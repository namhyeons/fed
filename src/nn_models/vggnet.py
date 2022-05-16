import torch
import torch.nn as nn
import math
from typing import Type, Any, Callable, Union, List, Optional

class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.GroupNorm(num_groups=16, num_channels=num_channels, eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

class MyBatchNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyBatchNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True, norm_layer=MyGroupNorm):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class VGG(nn.Module):

    def __init__(self, \
            norm_layer: Optional[Callable[..., nn.Module]] = None, num_classes = 10):

        super(VGG, self).__init__()

        # layer 1 
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3,6, kernel_size=5, stride=1,bias=False) # output size 28x28x6
        # self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2) # 14x14x6
        # layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1,bias=False) # 10x10x16
        #layer 3
        self.linear1 = nn.Linear(400,120)
        #layer 4
        self.linear2 = nn.Linear(120,84)
        #5
        self.linear3 = nn.Linear(84,84)
        #6
        self.linear4 = nn.Linear(84,256)
        #7
        self.linear5 = nn.Linear(256,10)
        

    def forward(self, x):
       #layer1
       x = self.conv1(x)
       x = self.relu(x)
       x = self.maxpool(x)
       #layer2
       x = self.conv2(x)
       x = self.relu(x)
       x = self.maxpool(x)
       x = x.view(-1,16*5*5)
       #layer3
       x = self.linear1(x)
       x = self.relu(x)
       #layer4
       x = self.linear2(x)
       x = self.relu(x)
       #layer5
       x = self.linear3(x)
       x = self.relu(x)
       #layer6
       x = self.linear4(x)
       #layer7
       x = self.linear5(x)
       return x


class MobileNetV2(nn.Module):

    def __init__(self, \
            norm_layer: Optional[Callable[..., nn.Module]] = None, num_classes = 10):

        super(MobileNetV2, self).__init__()

        # layer 1 
        self.norm_layer = norm_layer
        self.conv10 = nn.Conv2d(3,32, kernel_size=3, stride=1,padding=1,bias=False) 
        # block 1
        self.conv11 = nn.Conv2d(32,32, kernel_size=3, stride=1,padding=1,groups=32,bias=False) 
        self.conv12 = nn.Conv2d(32,16, kernel_size=1, stride=1,bias=False) 
        # block 2
        self.conv21 = nn.Conv2d(16,96, kernel_size=1, stride=1,bias=False)
        self.conv22 = nn.Conv2d(96,96, kernel_size=3, stride=1,padding=1,groups=96,bias=False) 
        self.conv23 = nn.Conv2d(96,24, kernel_size=1, stride=1,bias=False) 
      
        self.conv24 = nn.Conv2d(24,144, kernel_size=1, stride=1,bias=False) 
        self.conv25 = nn.Conv2d(144,144, kernel_size=3, stride=1,padding=1,groups=144,bias=False) 
        self.conv26 = nn.Conv2d(144,24, kernel_size=1, stride=1,padding=1,bias=False) 
        # block 3
        self.conv31 = nn.Conv2d(24,144, kernel_size=1, stride=1,bias=False)
        self.conv32 = nn.Conv2d(144,144, kernel_size=3, stride=2,padding=1,groups=144,bias=False) 
        self.conv33 = nn.Conv2d(144,32, kernel_size=1, stride=1,bias=False) 
      
        self.conv34 = nn.Conv2d(32,192, kernel_size=1, stride=1,bias=False) 
        self.conv35 = nn.Conv2d(192,192, kernel_size=3, stride=1,padding=1,groups=192,bias=False) 
        self.conv36 = nn.Conv2d(192,32, kernel_size=1, stride=1,bias=False) 

        self.conv37 = nn.Conv2d(32,192, kernel_size=1, stride=1,bias=False) 
        self.conv38 = nn.Conv2d(192,192, kernel_size=3, stride=1,padding=1,groups=192,bias=False) 
        self.conv39 = nn.Conv2d(192,32, kernel_size=1, stride=1,bias=False) 
        # block 4
        self.conv41 = nn.Conv2d(32,192, kernel_size=1, stride=1,bias=False)
        self.conv42 = nn.Conv2d(192,192, kernel_size=3, stride=2,padding=1,groups=192,bias=False) 
        self.conv43 = nn.Conv2d(192,64, kernel_size=1, stride=1,bias=False) 
      
        self.conv44 = nn.Conv2d(64,384, kernel_size=1, stride=1,bias=False) 
        self.conv45 = nn.Conv2d(384,384, kernel_size=3, stride=1,padding=1,groups=384,bias=False) 
        self.conv46 = nn.Conv2d(384,64, kernel_size=1, stride=1,bias=False) 

        self.conv47 = nn.Conv2d(64,384, kernel_size=1, stride=1,bias=False) 
        self.conv48 = nn.Conv2d(384,384, kernel_size=3, stride=1,padding=1,groups=384,bias=False) 
        self.conv49 = nn.Conv2d(384,64, kernel_size=1, stride=1,bias=False) 

        self.conv410 = nn.Conv2d(64,384, kernel_size=1, stride=1,bias=False) 
        self.conv411 = nn.Conv2d(384,384, kernel_size=3, stride=1,padding=1,groups=384,bias=False) 
        self.conv412 = nn.Conv2d(384,64, kernel_size=1, stride=1,bias=False) 
        # block 5
        self.conv51 = nn.Conv2d(64,384, kernel_size=1, stride=1,bias=False)
        self.conv52 = nn.Conv2d(384,384, kernel_size=3, stride=1,padding=1,groups=384,bias=False) 
        self.conv53 = nn.Conv2d(384,96, kernel_size=1, stride=1,bias=False) 
      
        self.conv54 = nn.Conv2d(96,576, kernel_size=1, stride=1,bias=False) 
        self.conv55 = nn.Conv2d(576,576, kernel_size=3, stride=1,padding=1,groups=576,bias=False) 
        self.conv56 = nn.Conv2d(576,96, kernel_size=1, stride=1,bias=False) 

        self.conv57 = nn.Conv2d(96,576, kernel_size=1, stride=1,bias=False) 
        self.conv58 = nn.Conv2d(576,576, kernel_size=3, stride=1,padding=1,groups=576,bias=False) 
        self.conv59 = nn.Conv2d(576,96, kernel_size=1, stride=1,bias=False) 
        # block 6
        self.conv61 = nn.Conv2d(96,576, kernel_size=1, stride=1,bias=False)
        self.conv62 = nn.Conv2d(576,576, kernel_size=3, stride=2,padding=1,groups=576,bias=False) 
        self.conv63 = nn.Conv2d(576,160, kernel_size=1, stride=1,bias=False) 
      
        self.conv64 = nn.Conv2d(160,960, kernel_size=1, stride=1,bias=False) 
        self.conv65 = nn.Conv2d(960,960, kernel_size=3, stride=1,padding=1,groups=960,bias=False) 
        self.conv66 = nn.Conv2d(960,160, kernel_size=1, stride=1,bias=False)

        self.conv67 = nn.Conv2d(160,960, kernel_size=1, stride=1,bias=False) 
        self.conv68 = nn.Conv2d(960,960, kernel_size=3, stride=1,padding=1,groups=960,bias=False) 
        self.conv69 = nn.Conv2d(960,160, kernel_size=1, stride=1,bias=False)  
        # block 7
        self.conv71 = nn.Conv2d(160,960, kernel_size=1, stride=1,bias=False)
        self.conv72 = nn.Conv2d(960,960, kernel_size=3, stride=1,padding=1,groups=960,bias=False) 
        self.conv73 = nn.Conv2d(960,320, kernel_size=1, stride=1,bias=False) 

        self.conv8  = nn.Conv2d(320,1280, kernel_size=1, stride=1,bias=False) 
        self.avgPool = nn.AvgPool2d(kernel_size=4)
        
        self.linear9 = nn.Linear(1280,100)

      


        

    def forward(self, x):
        
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
       
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        x = self.conv35(x)
        x = self.conv36(x)
        x = self.conv37(x)
        x = self.conv38(x)
        x = self.conv39(x)


        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.conv45(x)
        x = self.conv46(x)
        x = self.conv47(x)
        x = self.conv48(x)
        x = self.conv49(x)
        x = self.conv410(x)
        x = self.conv411(x)
        x = self.conv412(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        x = self.conv55(x)
        x = self.conv56(x)
        x = self.conv57(x)
        x = self.conv58(x)
        x = self.conv59(x)

        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        x = self.conv64(x)
        x = self.conv65(x)
        x = self.conv66(x)
        x = self.conv67(x)
        x = self.conv68(x)
        x = self.conv69(x)

        x = self.conv71(x)
        x = self.conv72(x)
        x = self.conv73(x)

        x = self.conv8(x)
        x = self.avgPool(x)

        x = x.view(-1,1280)

        x = self.linear9(x)
        
        return x
     
       
       
       
       
       
       
       

       
     

def make_VGG(norm='gn'):
    if norm == 'gn':
        norm_layer = MyGroupNorm
        
    elif norm == 'bn':
        norm_layer = MyBatchNorm

    return VGG(norm_layer=norm_layer)
    
def make_MobileNetV2(norm='gn'):
    if norm == 'gn':
        norm_layer = MyGroupNorm
        
    elif norm == 'bn':
        norm_layer = MyBatchNorm

    return MobileNetV2(norm_layer=norm_layer)
    

if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    model = make_VGG(n_blocks=4, norm='bn')

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                print_per_layer_stat=False, verbose=True, units='MMac')

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

