import torch
from grid import *
from graph import build_groups
from integral_weight import WeightsParameterization
from integral_weight import InterpolationWeights1D
from integral_weight import InterpolationWeights2D
from utils import get_module_by_name
from utils import get_parent_name
from utils import fuse_batchnorm
from quadrature import TrapezoidalQuadrature
from torch.nn.utils.parametrize import register_parametrization
from permutation import NOptPermutation


class IntegralGroup(torch.nn.Module):
    def __init__(self, grid_1d, parameterizations):
        super(IntegralGroup, self).__init__()
        self.grid_1d = grid_1d
        self.parameterizations = parameterizations
        self.reset_grid(grid_1d)

    def get_grid(self):
        return self.grid_1d

    def reset_grid(self, grid_1d):
        self.grid_1d = grid_1d

        for obj, dim in self.parameterizations:
            obj.grid.reset_grid(dim, grid_1d)

    def resize(self, new_size):
        if hasattr(self.grid_1d, 'resize'):
            self.grid_1d.resize(new_size)

    def count_elements(self):
        num_el = 0

        for p, dim in self.parameterizations:
            weight = p(None)
            num_el += weight.numel()

        return num_el


class IntegralModel(torch.nn.Module):
    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        self.groups = torch.nn.ModuleList(groups)

    def forward(self, x):
        for group in self.groups:
            group.get_grid().generate_grid()

        return self.model(x)

    def resize(self, sizes):
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids_1d):
        for group, grid_1d in zip(self.groups, grids_1d):
            group.reset_grid(grid_1d)

    def count_elements(self):
        return [
            group.count_elements() for group in self.groups
        ]


class IntegralWrapper:
    def __init__(self, fuse_bn=True, init_from_discrete=True,
                 optimize_iters=100, start_lr=1e-2,
                 **permutation_config):

        self.rearranger = NOptPermutation(**permutation_config)
        self.init_from_discrete = init_from_discrete
        self.fuse_bn = fuse_bn
        self.optimize_iters = optimize_iters
        self.start_lr = start_lr

    def wrap_module(self, module, name):
        quadrature = None
        func = None

        if 'weight' in name:
            weight = getattr(module, name)
            cont_shape = weight.shape[:2]

            if len(weight.shape) > 2:
                discrete_shape = weight.shape[2:]
            else:
                discrete_shape = None

            func = InterpolationWeights2D(cont_shape, discrete_shape)
            quadrature = TrapezoidalQuadrature([1])

        elif 'bias' in name:
            bias = getattr(module, name)
            cont_shape = bias.shape[0]
            func = InterpolationWeights1D(cont_shape)

        return func, quadrature

    def wrap_model(self, model, example_input):

        if self.fuse_bn:
            model.eval()
            model = fuse_batchnorm(model)

        groups = build_groups(model, example_input)

        if self.init_from_discrete and self.rearranger is not None:
            for i, group in enumerate(groups):
                print(f'Rearranging of group {i}')
                self.rearranger.permute(
                    group['params'], group['size']
                )

        integral_groups = []

        for group in groups:
            min_val = max(3, group['size'] // 2)
            max_val = group['size']
            distrib = UniformDistribution(min_val, max_val)
            grid_1d = RandomUniformGrid1D(distrib)
            group['grid'] = grid_1d

        for group in groups:
            parameterizations = []

            for p in group['params']:
                parent_name, name = get_parent_name(p['name'])

                if parent_name != '':
                    parent = get_module_by_name(model, parent_name)
                else:
                    parent = model

                if not hasattr(parent, 'parametrizations') \
                        or name not in parent.parametrizations:
                    func, quadrature = self.wrap_module(parent, name)
                    g_lst = [
                        g['grid'] for g in p['value'].grids
                        if g is not None
                    ]
                    delattr(p['value'], 'grids')
                    grid = GridND(*g_lst)
                    parameterization = WeightsParameterization(
                        func, grid, quadrature
                    ).to(p['value'].device)
                    target = torch.clone(p['value'])
                    register_parametrization(
                        parent, name, parameterization, unsafe=True
                    )
                    if self.init_from_discrete:
                        optimize_parameters(
                            parent, name, target,
                            self.start_lr, self.optimize_iters
                        )

                else:
                    parameterization = parent.parametrizations[name][0]

                parameterizations.append([parameterization, p['dim']])

            integral_groups.append(
                IntegralGroup(group['grid'], parameterizations)
            )

        integral_model = IntegralModel(model, integral_groups)

        return integral_model


def optimize_parameters(module, attr, target,
                        start_lr=1e-2, iterations=100):
    module.train()
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(module.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=iterations // 5, gamma=0.2
    )

    for i in range(iterations):
        weight = getattr(module, attr)
        loss = criterion(weight, target)
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

        if i == 0:
            print('loss before optimization: ', loss)
        if i == iterations - 1:
            print('loss after optimization: ', loss)


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18().cuda()
    sample_shape = [1, 3, 100, 100]
    wrapper = IntegralWrapper(init_from_discrete=False, fuse_bn=True)
    integral_model = wrapper.wrap_model(model, sample_shape)
    print("Group sizes: ", integral_model.count_elements())
    integral_model(torch.rand(sample_shape).cuda())