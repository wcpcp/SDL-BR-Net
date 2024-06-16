import torch
import torch.nn as nn
from .utils import InitWeights_He
import time

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=False))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x = self.down(x)
        return x

# class up(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(up, self).__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
#                                padding=0, stride=2, bias=False),   #或者换成双线性插值
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.1, inplace=False))

#     def forward(self, x):
#         x = self.up(x)
#         return x

class up(nn.Module):
    def __init__(self, in_c, out_c):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class rr(nn.Module):
    def __init__(self, in_c, out_c):
        super(rr, self).__init__()
        # self.down = down(in_c, in_c)
        self.weight_map = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.over_pixel = DSConv3x3(in_c, in_c, dilation=2)
        self.down_over_pixel1 = DSConv3x3(in_c, in_c, dilation=2)
        self.down_over_pixel2 = DSConv3x3(in_c, in_c, dilation=2)
        
        self.up0 = up(in_c, in_c)
        self.up1 = up(in_c, in_c)
        self.brrelu = nn.Sequential(
            nn.Conv2d(3*in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False)
        )

        self.down_conv = down(in_c, in_c)
        self.down_mean = nn.AvgPool2d(kernel_size=2, stride=2)

    #now the best 
#     def forward(self, x):

#         x_up = self.over_pixel(x)
#         x_map = self.weight_map(x_up)
#         x_up = x_map * x_up

#         x_down_0 = self.down(x)
#         x_down_1 = self.down(x_up)
        
#         x_down = x_down_0 + x_down_1

#         x_down = self.down_over_pixel(x_down)

#         x_res = self.up0(x_down)

#         x_out = self.brrelu(torch.cat([x_res, x_up], dim=1))

#         return x_out

    #17号
    def forward(self, x):

        x_sps = self.over_pixel(x)
        x_map = self.weight_map(x_sps)
        x_sps = x_map * x_sps

        x_down_conv = self.down_conv(x_sps)
        x_down_mean = self.down_mean(x_sps)

        x_down_conv = self.down_over_pixel1(x_down_conv)
        x_down_mean = self.down_over_pixel2(x_down_mean)
        
        x_down_conv = self.up0(x_down_conv)
        x_down_mean = self.up1(x_down_mean)

        x_out = self.brrelu(torch.cat([x_sps, x_down_conv, x_down_mean], dim=1))

        return x_out


class fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(fusion, self).__init__()
        self.weight_map1 = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.weight_map2 = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.weight_map3 = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.weight_map4 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dw1 = DSConv3x3(out_c, out_c, dilation=1)
        self.dw2 = DSConv3x3(out_c, out_c, dilation=2)
        self.dw3 = DSConv3x3(out_c, out_c, dilation=3)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3*in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(in_c),
        #     nn.ReLU(inplace=False)
        # )
        
        self.pre = nn.Sequential(
            nn.Conv2d(4*in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.Sigmoid()
        )
        
        self.attention = ChannelAttention(4*in_c)
    
    def forward(self, F):
        
        # F = self.conv(torch.cat([F3, F4, F5], dim=1))
        f1 = self.dw1(F)
        f1_w = self.weight_map1(f1)
        f2 = self.dw2(F)
        f2_w = self.weight_map2(f2)
        f3 = self.dw3(F)
        f3_w = self.weight_map3(f3)
        
        f1_s = f1 * f1_w
        f2_s = f2 * f2_w
        f3_s = f3 * f3_w
        
        F_s = F * self.weight_map4(torch.cat([f1_w, f2_w, f3_w], dim=1))
        
        f_fusion = torch.cat([F_s, f1_s, f2_s, f3_s], dim=1)
        
        f_out = f_fusion * self.attention(f_fusion)
        
        out = self.pre(f_out)
        
        return out
        
        

# Remove redundancy
class RMR(nn.Module):
    def __init__(self, in_c=32, out_c=32):
        super(RMR, self).__init__()
        self.rr = rr(in_c, out_c)
        # self.rr4 = rr(in_c, out_c)
        # self.rr5 = rr(in_c, out_c)
        # self.dw1 = DSConv3x3(out_c, out_c, dilation=1)
        # self.dw2 = DSConv3x3(out_c, out_c, dilation=2)
        # self.dw3 = DSConv3x3(out_c, out_c, dilation=3)
        # self.pre = nn.Sequential(
        #     nn.Conv2d(4*out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_c),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        #     # nn.Sigmoid()
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        
        
        self.fusion = fusion(in_c, out_c)
        self.sigmoid = nn.Sigmoid()
        
        
        self.apply(InitWeights_He)


    def forward(self, pre1=None, img=None):
        pre1 = self.sigmoid(pre1)
        input_ = torch.cat([pre1,img], dim=1)
        
        x3 = self.conv3(input_)
        x5 = self.conv5(x3)
        x7 = self.conv7(x3 + x5)
        
        F5_ = self.rr(x7)

        out = self.fusion(F5_)
        
        return out

