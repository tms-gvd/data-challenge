from base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MnistModel(BaseModel):
    def __init__(self, nb_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nb_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CustomResNet18(ResNet):
    def __init__(self, nb_classes):
        super(CustomResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], 1000)
        weights = ResNet18_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, nb_classes)

    def forward(self, x, debug=False):

        x = self.conv1(x)
        if debug:
            print(x.shape)

        x = self.bn1(x)
        if debug:
            print(x.shape)

        x = self.relu(x)
        if debug:
            print(x.shape)

        x = self.maxpool(x)
        if debug:
            print(x.shape)

        x = self.layer1(x)
        if debug:
            print(x.shape)

        x = self.layer2(x)
        if debug:
            print(x.shape)

        x = self.layer3(x)
        if debug:
            print(x.shape)

        x = self.layer4(x)
        if debug:
            print(x.shape)

        x = self.avgpool(x)
        if debug:
            print(x.shape)

        x = torch.flatten(x, 1)
        if debug:
            print(x.shape)

        x = self.fc(x)
        if debug:
            print(x.shape)

        return x


class CustomResNet50(ResNet):
    def __init__(self, nb_classes):
        super(CustomResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], 1000)
        weights = ResNet50_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, nb_classes)

    def forward(self, x, debug=False):

        x = self.conv1(x)
        if debug:
            print(x.shape)

        x = self.bn1(x)
        if debug:
            print(x.shape)

        x = self.relu(x)
        if debug:
            print(x.shape)

        x = self.maxpool(x)
        if debug:
            print(x.shape)

        x = self.layer1(x)
        if debug:
            print(x.shape)

        x = self.layer2(x)
        if debug:
            print(x.shape)

        x = self.layer3(x)
        if debug:
            print(x.shape)

        x = self.layer4(x)
        if debug:
            print(x.shape)

        x = self.avgpool(x)
        if debug:
            print(x.shape)

        x = torch.flatten(x, 1)
        if debug:
            print(x.shape)

        x = self.fc(x)
        if debug:
            print(x.shape)

        return x
