import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLPBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(MLPBlock, self).__init__()
        self.l1 = nn.Linear(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.l2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetMLP(nn.Module):

    def __init__(self, block, layers, inplanes, num_classes=1000):
        self.inplanes = inplanes
        super(ResNetMLP, self).__init__()
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes, layers[1])
        self.layer3 = self._make_layer(block, inplanes, layers[2])

        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(inplanes, num_classes)
        else:
            self.fc = nn.Linear(inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # EXPANSION FOR NOW IS 1
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x4 = self.layer4(x3)

        x = self.fc(x3)

        return x  # output, intermediate



def resmlp13( **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMLP(MLPBlock, [2, 2, 2], **kwargs)

    return model

