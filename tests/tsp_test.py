import torch
import time
import torch_integral
from torch_integral.tsp_solver import two_opt_find_permutation


tensors = [
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0},
	{'value': torch.randn(512, 512, 3, 3), 'dim': 0}
]

t = time.time()
ind = two_opt_find_permutation(tensors, 64, 100, 0.01)
t = time.time() - t

print("time: ", t)
