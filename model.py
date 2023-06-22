from torch.nn import Module
from torch import nn
from rational import *

class Model(Module):
    def __init__(self, ActivationType):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        if ActivationType == 1:
            self.R1 = Rational()
        elif ActivationType == 2:
            self.R1 = nn.ELU()
        else:
            self.R1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if ActivationType == 1:
            self.R2 = Rational()
        elif ActivationType == 2:
            self.R2 = nn.ELU()
        else:
            self.R2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        if ActivationType == 1:
            self.R3 = Rational()
        elif ActivationType == 2:
            self.R3 = nn.ELU()
        else:
            self.R3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        if ActivationType == 1:
            self.R4 = Rational()
        elif ActivationType == 2:
            self.R4 = nn.ELU()
        else:
            self.R4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        if ActivationType == 1:
            self.R5 = Rational()
        elif ActivationType == 2:
            self.R5 = nn.ELU()
        else:
            self.R5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.R1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.R2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.R3(y)
        y = self.fc2(y)
        y = self.R4(y)
        y = self.fc3(y)
        y = self.R5(y)
        return y