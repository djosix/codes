import torch
from torch.autograd import Variable
from torch.optim import SGD

m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)

a = Variable(m1, requires_grad=True)
b = Variable(m2, requires_grad=True)

optimizer = SGD([a, b], lr=0.06)

for _ in range(10):
    loss = ((a @ b.transpose(0, 1)) ** 2).sum()

    optimizer.zero_grad()   # clear gradient buffer
    loss.backward()         # compute gradient
    optimizer.step()        # update parameters
    
    print('loss:', loss.data[0])
