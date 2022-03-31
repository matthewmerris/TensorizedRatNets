from torch.nn import Module
from torch import nn
from rational import *

# modified to be styled after github.com/erykml/medium_articles

class Model(Module):
    def __init__(self, n_classes, UseRational=False, UseRELU=False):
        super(Model, self).__init__()
        
        # Determine activation to use
        if UseRational:
            activation = Rational()
        else if UseRELU:
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
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(x)
        
        return logits, probs
