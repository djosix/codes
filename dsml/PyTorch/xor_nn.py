import numpy as np
import torch as T
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sgd = T.optim.Adagrad(self.parameters(), lr=0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def evaluate(self, x):
        self.eval()
        return self.forward(x)

    def fit(self, x, y):
        self.train()
        self.zero_grad()
        ((self.forward(x) - y) ** 2).sum().backward()
        self.sgd.step()
        

X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
] * 100

Y_train = [
    [0],
    [1],
    [1],
    [0]
] * 100


X = V(T.Tensor(X_train))
Y = V(T.Tensor(Y_train))
n = Net()

E = []
for i in range(1000):
    e = ((n.evaluate(X) - Y)**2).sum().data[0]
    E += [e]
    print(e)
    if e < 5: break
    n.fit(X, Y)
print((((n.evaluate(X)>0.5).int() == Y.int()).int().sum().data[0] / len(X)))

import matplotlib.pyplot as plt
plt.plot(range(len(E)), E)
plt.show()

# for i in range(100):
