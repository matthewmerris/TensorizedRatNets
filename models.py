from torch.nn import Module
from torch import nn
from rational import *


class LinearBlock(nn.Module):
    def __init__(self, n_dim_in: int, n_dim_out: int):
        super().__init__()
        self.linear = nn.Linear(n_dim_in, n_dim_out)
        self.rat = Rational()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.rat(y)
        return y

class LinearOut(nn.Module):
    def __init__(self, n_dim_in: int, n_dim_out: int):
        super().__init__()
        self.linear = nn.Linear(n_dim_in, n_dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y

class LenetLinear(Module):
    def __init__(self, UseRational=False, UseRELU=False):
        super(LenetLinear, self).__init__()

        # Determine activation to use
        if UseRational:
            activation = Rational()
        elif UseRELU:
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()


        # Alternate approach, using ModuleDict
        self.layers = nn.ModuleDict({
            'layer_0' : LinearBlock(20*20, 10),
            'layer_1' : LinearOut(10,10)
        })


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers.values():
            x = layer(x)
        # breakpoint()

        return x

class Lenet300(Module):
    def __init__(self, UseRational=False, UseRELU=False):
        super(Lenet300, self).__init__()

        # Determine activation to use
        if UseRational:
            activation = Rational()
        elif UseRELU:
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()


        # Alternate approach, using ModuleDict
        self.layers = nn.ModuleDict({
            'layer_0' : LinearBlock(20*20, 300),
            'layer_1' : LinearOut(300,10)
        })


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers.values():
            x = layer(x)
        # breakpoint()

        return x

class Lenet300100(Module):
    def __init__(self, UseRational=False, UseRELU=False):
        super(Lenet300100, self).__init__()

        # Determine activation to use
        if UseRational:
            activation = Rational()
        elif UseRELU:
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        # Define the feature extractor
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(28*28, 300),
        #     activation,
        #     nn.Linear(300,100),
        #     activation,
        #     nn.Linear(100,n_classes)
        # )

        # Alternate approach, using ModuleDict
        self.layers = nn.ModuleDict({
            'layer_0' : LinearBlock(28*28, 300),
            'layer_1' : LinearBlock(300,100),
            'layer_2' : LinearOut(100,10)
        })


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers.values():
            x = layer(x)
        # breakpoint()

        return x

## Repeat architectural setup for Lenet5 (i.e. convolutional block set up)
class Conv2dBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ker_sz: int, stride_sz: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_chan,out_channels=out_chan, kernel_size=ker_sz, stride=stride_sz)
        self.rat = Rational()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.rat(y)
        return y

class Lenet5(Module):
    def __init__(self, n_classes, UseRational=False, UseRELU=False):
        super(Lenet5, self).__init__()

        # Determine activation to use
        if UseRational:
            activation = Rational()
        elif UseRELU:
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        # Define the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            activation,
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            activation,
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            activation
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            activation,
            nn.Linear(in_features=84, out_features=n_classes)
        )

        self.layers = nn.ModuleDict({
            'layer_0' : Conv2dBlock(in_chan=1, out_chan=6, ker_sz=5, stride_sz=1),
            'layer_1' : nn.AvgPool2d(kernel_size=2),
            'layer_2' : Conv2dBlock(in_chan=6, out_chan=16,ker_sz=5, stride_sz=1),
            'layer_3' : nn.AvgPool2d(kernel_size=2),
            'layer_4' : Conv2dBlock(in_chan=16, out_chan=120, ker_sz=5, stride_sz=1),
            'layer_5' : LinearBlock(n_dim_in=120, n_dim_out=84),
            'layer_6' : nn.Linear(in_features=84, out_features=10)
        })

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(x)

        return logits, probs