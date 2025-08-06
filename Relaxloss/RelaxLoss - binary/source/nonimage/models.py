import torch
import torch.nn as nn


class TexasClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=0):
        super(TexasClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6169, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        if droprate > 0:
            self.classifier = nn.Sequential(nn.Dropout(droprate),
                                            nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=0):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        if droprate > 0:
            self.classifier = nn.Sequential(nn.Dropout(droprate),
                                            nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


class CovertypeClassifier(nn.Module):
    def __init__(self, input_size=54, num_classes=7, droprate=0.1):
        super(CovertypeClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdultClassifier(nn.Module):
    def __init__(self, input_size=108, binary=True, droprate=0.3):
        super(AdultClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        out_dim = 1 if binary else 2
        self.classifier = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def Purchase(**kwargs):
    return PurchaseClassifier(**kwargs)

def Texas(**kwargs):
    return TexasClassifier(**kwargs)

def Covertype(**kwargs):
    return CovertypeClassifier(**kwargs)

def Adult(**kwargs):
    return AdultClassifier(**kwargs)
