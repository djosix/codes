import torch

m = torch.ones(2, 3)
print(m)

m = torch.zeros(2, 3)
print(m)

m = torch.eye(2, 3)
print(m)

m = torch.rand(2, 3)
print(m)

m = torch.randn(2, 3)
print(m)

m1 = torch.ones(3, 3)
m2 = torch.zeros(3, 3)

m = torch.cat([m1, m2], 0)
print(m)

m = torch.cat([m1, m2], 1)
print(m)

m = torch.stack([m1, m2], 0)
print(m)

import numpy as np

a = np.arange(5)
print(a)

t = torch.from_numpy(a)
print(t)

a[0] = 5
print(t)

print(t.numpy())

try:
    print('cuda:', t.cuda())
except:
    print('cpu:', t.cpu())
