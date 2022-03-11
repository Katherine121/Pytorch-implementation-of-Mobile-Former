import numpy as np
import torch
import torch.nn as nn


# inputs: b c h w
# output: b c h w
class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.stride = stride
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.ac1 = nn.Hardtanh(min_val=0, max_val=6)

        self.conv2 = nn.Conv2d(hid, hid, kernel_size=ks, stride=stride,
                               padding=ks // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.ac2 = nn.Hardtanh(min_val=0, max_val=6)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, x, z):
        # b*hid*h*w
        out = self.bn1(self.conv1(x))
        out = self.ac1(out)

        # b*hid*h*w
        out = self.bn2(self.conv2(out))
        out = self.ac2(out)

        out = self.bn3(self.conv3(out))

        if self.se is not None:
            out = self.se(out)
        # 如果图片没有下采样，则残差连接，此模块没有下采样所以要残差连接
        if self.stride == 1:
            out = out + self.shortcut(x)
        return out


# inputs: b c h w
# output: b c h w
class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.stride = stride
        self.se = se

        self.dw_conv1 = nn.Conv2d(inp, hid, kernel_size=ks, stride=stride,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_ac1 = nn.Hardtanh(min_val=0, max_val=6)

        self.pw_conv1 = nn.Conv2d(hid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(inp, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_ac2 = nn.Hardtanh(min_val=0, max_val=6)

        self.pw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, x, z):
        # b*hid*h*w
        out = self.dw_bn1(self.dw_conv1(x))
        out = self.dw_ac1(out)

        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        # b*hid*h*w
        out = self.dw_bn2(self.dw_conv2(out))
        out = self.dw_ac2(out)

        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        # 如果图片没有下采样，则残差连接，此模块有下采样所以不残差连接
        if self.stride == 1:
            out = out + self.shortcut(x)
        return out
