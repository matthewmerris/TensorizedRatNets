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

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(x)

        return logits, probs

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
            'layer_2' : nn.Linear(100,10)
        })


    def forward(self, x):
        # x = self.feature_extractor(x)
        for layer in self.layers.values():
            x = layer(x)

        return x
