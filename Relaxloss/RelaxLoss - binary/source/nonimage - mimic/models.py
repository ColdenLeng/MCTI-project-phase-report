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


class SparseMIMICClassifier(nn.Module):
    def __init__(self, input_size=8390, num_classes=100, droprate=0.5):
        super(SparseMIMICClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),     # 🔹缩减维度
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
        
'''
     #def __init__(self, input_size=100, num_classes=10, droprate=0):   
class MIMICClassifier(nn.Module):
    def __init__(self, input_size=24, num_classes=70, droprate=0.3):
        super(MIMICClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        if droprate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(droprate),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
'''
'''
class MIMICClassifier(nn.Module):
    def __init__(self, input_size=24, num_classes=70, droprate=0.3):
        super(MIMICClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
        
'''
'''
a fully connected feedforward neural network designed for structured tabular data. 
It takes 24 input features and outputs predictions over 70 classes. 
The model consists of four hidden layers with increasing depth and batch normalization, 
followed by a dropout layer for regularization and a final classification layer.
'''
import torch.nn as nn

class MIMICClassifier(nn.Module):
    def __init__(self, input_size=3605, num_classes=2, droprate=0.3):
        super(MIMICClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        return self.net(x)


'''
What is this model do, what thing they identify. Explain how the relaxloss work with this model, 
how they connected, what is the mode f is used for, how do we use the model f in the project, 
Implemented attack defence and train the model.
'''
def Purchase(**kwargs):
    return PurchaseClassifier(**kwargs)


def Texas(**kwargs):
    return TexasClassifier(**kwargs)
    
    
def MIMIC(**kwargs):
    return MIMICClassifier(**kwargs)
