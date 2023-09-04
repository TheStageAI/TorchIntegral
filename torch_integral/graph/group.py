import torch


class RelatedGroup(torch.nn.Module):
    """
    Class for grouping tensors and parameters.
    Group is a collection of paris of tensor and it's dimension.
    Two parameter tensors are considered to be in the same group
    if they should have the same integration grid.
    Group can contain subgroups. This means that parent group's grid is a con
    catenation of subgroups grids.

    Parameters
    ----------
    size: int.
        Each tensor in the group should have the same size along certain dimension.
    """

    def __init__(self, size):
        super(RelatedGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.params = []
        self.tensors = []
        self.operations = []

    def forward(self):
        pass

    def copy_attributes(self, group):
        self.size = group.size
        self.subgroups = group.subgroups
        self.parents = group.parents
        self.params = group.params
        self.tensors = group.tensors
        self.operations = group.operations

        for parent in self.parents:
            if group in parent.subgroups:
                i = parent.subgroups.index(group)
                parent.subgroups[i] = self

        if self.subgroups is not None:
            for sub in self.subgroups:
                if group in sub.parents:
                    i = sub.parents.index(group)
                    sub.parents[i] = self

        for param in self.params:
            param["value"].related_groups[param["dim"]] = self

        for tensor in self.tensors:
            tensor["value"].related_groups[tensor["dim"]] = self

    def append_param(self, name, value, dim, operation=None):
        """
        Adds parameter tensor to the group.

        Parameters
        ----------
        name: str.
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.params.append(
            {"value": value, "name": name, "dim": dim, "operation": operation}
        )

    def append_tensor(self, value, dim, operation=None):
        """
        Adds tensor to the group.

        Parameters
        ----------
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.tensors.append({"value": value, "dim": dim, "operation": operation})

    def clear_params(self):
        self.params = []

    def clear_tensors(self):
        self.tensors = []

    def set_subgroups(self, groups):
        self.subgroups = groups

        for subgroup in self.subgroups:
            if subgroup is not None:
                subgroup.parents.append(self)

    def build_operations_set(self):
        """Builds set of operations in the group."""
        self.operations = set([t["operation"] for t in self.tensors])

    def count_parameters(self):
        ans = 0

        for p in self.params:
            ans += p["value"].numel()

        return ans

    def __str__(self):
        result = ""

        for p in self.params:
            result += p["name"] + ": " + str(p["dim"]) + "\n"

        return result

    @staticmethod
    def append_to_groups(tensor, operation=None):
        attr_name = "related_groups"

        if hasattr(tensor, attr_name):
            for i, g in enumerate(getattr(tensor, attr_name)):
                if g is not None:
                    g.append_tensor(tensor, i, operation)
