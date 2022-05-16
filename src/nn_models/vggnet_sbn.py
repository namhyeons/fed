import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

class MyBatchNorm(nn.Module):
    def __init__(self, num_channels, track=True):
        super(MyBatchNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=track)
    
    def forward(self, x):
        x = self.norm(x)
        return x

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

class VGG(nn.Module):

    def __init__(self, num_classes, model_rate, track=True, scale=True):

        super(VGG, self).__init__()
        
        self.scaler = Scaler(model_rate, scale)
        self.inplanes = int(64 * model_rate)
        self.norm_layer = MyBatchNorm

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(self.inplanes, track)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    
        self.conv2 = nn.Conv2d(int(64 * model_rate), int(128 * model_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self.norm_layer(int(128 * model_rate), track)
    
        
        self.conv3 = nn.Conv2d(int(128 * model_rate), int(256 * model_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = self.norm_layer(int(256 * model_rate), track)

        self.conv4 = nn.Conv2d(int(256 * model_rate), int(512 * model_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = self.norm_layer(int(512 * model_rate), track)
    
        self.scala = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(512 * model_rate), 10)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.scaler(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.scaler(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.scaler(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = self.scaler(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.scala(x)
    
        x = x.view(x.size(0), -1)
    
        output = self.fc(x)
    
        return output

def make_VGG(num_class, model_rate, track=True, scale=True):
    
    return VGG(num_class, model_rate, track, scale)
    

if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    model = make_VGG(10, 4/4)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                print_per_layer_stat=False, verbose=True, units='MMac')

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

