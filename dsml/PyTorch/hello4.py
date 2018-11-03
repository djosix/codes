import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
       

model = Model()

model.parameters()

# y = model(INPUT)

model.train()
model.eval()

if torch.cuda.is_available():
    model.gpu()

model.cpu()

# torch.save(model.state_dict(), PATH)
# model.load_state_dict(torch.load(PATH))
