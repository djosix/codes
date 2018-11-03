import torch
from torch.autograd import Variable

m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)
a = Variable(m1)
b = Variable(m2)
c = a + b
print(c)