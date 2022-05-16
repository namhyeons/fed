import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np

class Scaler(nn.Module):
    def __init__(self, rate, scale):
        super().__init__()
        if scale:
            self.rate = rate
        else:
            self.rate = 1

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class MyBatchNorm(nn.Module):
    def __init__(self, num_channels, track):
        super(MyBatchNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=track)
    
    def forward(self, x):
        x = self.norm(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,  scaler_rate=1, downsample=None, track=True, scale=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.scaler = Scaler(scaler_rate, scale)
        self.bn1 = MyBatchNorm(planes, track)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MyBatchNorm(planes, track)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.scaler(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.scaler(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        output += residual
        output = self.relu(output)
        return output


class Multi_ResNet(nn.Module):
    """Resnet model
    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, hidden_size, block, layers, num_classes, scaler_rate, track, scale):

        super(Multi_ResNet, self).__init__()
        
        self.inplanes = hidden_size[0]
        self.norm_layer = MyBatchNorm
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.scaler = Scaler(scaler_rate, scale)
        self.bn1 = self.norm_layer(self.inplanes, track)

        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, hidden_size[0], layers[0], scaler_rate = scaler_rate, track=track, scale=scale)
        self.layer2 = self._make_layer(block, hidden_size[1], layers[1], stride=2, scaler_rate = scaler_rate, track=track, scale=scale)
        self.layer3 = self._make_layer(block, hidden_size[2], layers[2], stride=2, scaler_rate = scaler_rate, track=track, scale=scale)
        self.layer4 = self._make_layer(block, hidden_size[3], layers[3], stride=2, scaler_rate = scaler_rate, track=track, scale=scale)
        self.fc = nn.Linear(hidden_size[3], num_classes)
        self.scala = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1, scaler_rate = 1, track=True, scale=True):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        norm_layer = self.norm_layer
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, scaler_rate = scaler_rate, downsample=downsample, track=track, scale=scale))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes, scaler_rate = scaler_rate, track=track, scale=scale))
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.scaler(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.scala(x).view(x.size(0), -1)
        out = self.fc(out)

        return out


def resnet18_tiny(num_class, model_rate=1, track=True, scale=True):

    classes_size = num_class

    hidden_size = [64, 128, 256, 512]
    hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_size]

    scaler_rate = model_rate

    return Multi_ResNet(hidden_size, BasicBlock, [2,2,2,2], num_classes=classes_size, scaler_rate=scaler_rate, track=track, scale=scale)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    model = resnet18_tiny(200, 4/4)
    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True,
                                                print_per_layer_stat=False, verbose=True, units='MMac')

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

