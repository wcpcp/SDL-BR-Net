import math
# from model.attention import CAM_Module,PAM_Module
import torch
from torchvision import models
import torch.nn as nn
from .resnet import resnet34
# import resnet
from torch.nn import functional as F
import torchsummary
from torch.nn import init
# import model.gap as gap
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
from .utils import InitWeights_He


class up(nn.Module):
    def __init__(self, in_c, out_c):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class DcomNet(nn.Module):
    def __init__(self,num_class=1):
        super(DcomNet,self).__init__()
        
        # self.up_4_d = up(512,256)
        self.up_3_d = up(256,128)
        self.up_2_d = up(128,64)
        self.up_1_d = up(64,64)
        self.up_0_d = up(32,32)


        # self.up_4_c = up(512, 256)
        self.up_3_c = up(256, 128)
        self.up_2_c = up(128, 64)
        self.up_1_c = up(64, 64)
        self.up_0_c = up(32, 32)

        self.up_4 = up(512, 256)
        self.up_3 = up(256, 128)
        self.up_2 = up(128, 64)
        self.up_1 = up(64, 32)
        self.up_0 = up(32, 32)

        self.convdgt4 = conv(512,256)
        self.convdgt3 = conv(256,128)
        self.convdgt2 = conv(128,64)
        self.convdgt1 = conv(128,32)

        self.convcgt4 = conv(512, 256)
        self.convcgt3 = conv(256, 128)
        self.convcgt2 = conv(128, 64)
        self.convcgt1 = conv(128, 32)

        self.convgt4 = conv(768, 256)
        self.convgt3 = conv(384, 128)
        self.convgt2 = conv(192, 64)
        self.convgt1 = conv(96, 32)

        self.convoutd = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.Sigmoid()
        )
        self.convoutc =nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.Sigmoid()
        )
        self.convouts = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.Sigmoid()
        )

        self.apply(InitWeights_He)

        self.backbone = resnet34(pretrained=True)

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        e1 = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(e1)

        e2 = self.backbone.layer1(x)  # 1/4   64
        e3 = self.backbone.layer2(e2)  # 1/8   128
        e4 = self.backbone.layer3(e3)  # 1/16   256
        e5 = self.backbone.layer4(e4)  # 1/32   512

        e5_up = self.up_4(e5)   #1/16  256
        #dgt
        d4 = self.convdgt4(torch.cat([e5_up,e4],dim=1))  #256
        d4_up = self.up_3_d(d4)    #  1/8   128
        d3 = self.convdgt3(torch.cat([d4_up,e3],dim=1))  #128
        d3_up = self.up_2_d(d3)    #  1/4   64
        d2 = self.convdgt2(torch.cat([d3_up, e2], dim=1)) #64
        d2_up = self.up_1_d(d2)    #  1/2   64
        d1 = self.convdgt1(torch.cat([d2_up, e1], dim=1))   #1/2  32
        d1_up = self.up_0_d(d1)    #  1/1   32
        d_out = self.convoutd(d1_up)

        # cgt
        c4 = self.convcgt4(torch.cat([e5_up, e4], dim=1))
        c4_up = self.up_3_c(c4)
        c3 = self.convcgt3(torch.cat([c4_up, e3], dim=1))
        c3_up = self.up_2_c(c3)
        c2 = self.convcgt2(torch.cat([c3_up, e2], dim=1))
        c2_up = self.up_1_c(c2)
        c1 = self.convcgt1(torch.cat([c2_up, e1], dim=1))
        c1_up = self.up_0_c(c1)    #  1/1   32
        c_out = self.convoutc(c1_up)

        # gt
        s4 = self.convgt4(torch.cat([e5_up, d4, c4], dim=1))   #768->256
        s4_up = self.up_3(s4)                                  #128
        s3 = self.convgt3(torch.cat([s4_up, d3, c3], dim=1))   #384->128
        s3_up = self.up_2(s3)                                  #64
        s2 = self.convgt2(torch.cat([s3_up, d2, c2], dim=1))   #192->64
        s2_up = self.up_1(s2)                                  #32
        s1 = self.convgt1(torch.cat([s2_up, d1, c1], dim=1))   #96->32
        s1_up = self.up_0(s1)                                  #  1/1   32 
        s_out = self.convouts(torch.cat([s1_up,c1_up,d1_up],dim=1)) 

        return d_out, c_out, s_out





