import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from super_image import Trainer, TrainingArguments
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from datasets import load_dataset
import sys
from catalyst import dl
sys.path.append('../../')
import torch_integral
from torch_integral.permutation import RandomPermutation
from torch_integral.permutation import NOptPermutationModified
from torch_integral.utils import base_continuous_dims
from torch_integral.utils import get_attr_by_name


def test(model, name='realesrgan_x4'):
    path_to_image = '../text2image/astronaut_rides_horse.png'
    image = Image.open(path_to_image).convert('RGB')
    sr_image = model.predict(image)
    sr_image.save(f'./{name}.png')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

cont_dims = {}
lst_names = [f'conv{i}' for i in range(2, 5)]

for name, module in model.model.named_modules():
    last_part = name.split('.')[-1]
    if last_part in lst_names:
        cont_dims[name + '.weight'] = [0, 1]
        cont_dims[name + '.bias'] = [0]
    elif last_part == 'conv5':
        cont_dims[name + '.weight'] = [1]
    elif last_part == 'conv1':
        cont_dims[name + '.weight'] = [0]
        cont_dims[name + '.bias'] = [0]

model.model = torch_integral.IntegralWrapper(
    init_from_discrete=True, fuse_bn=True, verbose=True,
    optimize_iters=0, permutation_iters=50,
    # permutation_config={'class': RandomPermutation}
    permutation_config={'class':NOptPermutationModified, 'iters':50}
).wrap_model(model.model, [1, 3, 32, 32], cont_dims)

discrete_model = model.model.transform_to_discrete()
integral_model = model.model


# DISTILLATION
class DistillationLoss(torch.nn.Module):
    def __init__(self, discrete_model, integral_model):
        super().__init__()

        def save_feature_map_hook(module, input, output):
            module.feature_map = output

        for model in (discrete_model, integral_model):
            for name, module in model.named_modules():
                if name.endswith('.conv5'):
                    module.register_forward_hook(
                        save_feature_map_hook
                    )

        self.discrete_model = discrete_model.cuda()
        self.integral_model = integral_model.cuda()
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        def save_inp(mod, inp, out):
            mod.input = inp[0]

        self.integral_model.register_forward_hook(save_inp)

    def forward(self, predict, target):
        self.discrete_model.eval()
        inp = self.integral_model.input
        target = self.discrete_model(inp)
        loss = self.loss_fn(predict, target)

        for name, module in self.discrete_model.named_modules():
            if 'conv5' in name:
                fm_true = module.feature_map.detach()
                integral_conv = get_attr_by_name(
                    self.integral_model, name
                )
                fm_integral = integral_conv.feature_map
                fm_loss = self.loss_fn(fm_integral, fm_true)
                loss = loss + fm_loss

        return loss


integral_model.eval()
test(model, name='realesrgan_x4_full')

for group in integral_model.groups[:276]:
    if group.grid_size() == 32:
        new_size = 24
        group.resize(new_size)

test(model, name='realesrgan_x4_pruned')
print('compressed: ', integral_model.calculate_compression())

# DATA
patch_size = 48
augmented_dataset = load_dataset(
    'eugenesiow/Div2k', 'bicubic_x4', split='train'
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset, patch_size)
eval_dataset = EvalDataset(
    load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
)

# TRAIN
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=integral_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

train_dataloader = trainer.get_train_dataloader()
eval_dataloader = trainer.get_eval_dataloader()
criterion = DistillationLoss(discrete_model, integral_model)
opt = torch.optim.Adam(
    integral_model.parameters(), lr=1e-3, weight_decay=0,
)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)

runner = dl.SupervisedRunner(
    input_key="features", output_key="output",
    target_key="targets", loss_key="loss"
)

callbacks = [
    dl.SchedulerCallback(
        mode='batch', loader_key='train', metric_key='loss'
    )
]

loggers = []
epochs = 1
log_dir = './logs/realesrgan'

with torch_integral.grid_tuning(integral_model, False, True):
    runner.train(
        model=integral_model,
        criterion=criterion,
        optimizer=opt,
        scheduler=sched,
        loaders={'train': train_dataloader},
        num_epochs=epochs,
        callbacks=callbacks,
        loggers=loggers,
        logdir=log_dir,
        # valid_loader="valid",
        # valid_metric="loss",
        # minimize_valid_metric=True,
        cpu=False,
        verbose=True,
        fp16=False
    )

test(model, name='realesrgan_x4_pruned')
