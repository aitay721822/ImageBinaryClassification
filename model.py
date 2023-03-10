import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.resnet as rs


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.resnet = models.resnet50(weights=rs.ResNet50_Weights.DEFAULT)
        
        self.classifcation = nn.Sequential(
            # first layer
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # second layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # output layer
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifcation(x)
        return x