import torch
import torch.nn as nn


class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, stride):
        super(Mobile, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(inp, inp // 2, kernel_size=ks, stride=stride,
                               padding=ks // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(inp // 2)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(inp // 2, out, kernel_size=ks, stride=stride,
                               padding=ks // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.ac2 = nn.ReLU()

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, x):
        # b*hid*h*w
        out = self.bn1(self.conv1(x))
        out = self.act1(out)

        # b*hid*h*w
        out = self.bn2(self.conv2(out))
        out = self.ac2(out)

        # 如果图片没有下采样，则残差连接，此模块没有下采样所以要残差连接
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, stride):
        super(MobileDown, self).__init__()
        self.stride = stride

        self.dw_conv1 = nn.Conv2d(inp, inp // 2, kernel_size=ks, stride=stride,
                                  padding=ks // 2, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(inp // 2)
        self.dw_act1 = nn.ReLU()

        self.pw_conv1 = nn.Conv2d(inp // 2, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(hid)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(out)
        self.dw_ac2 = nn.ReLU()

    def forward(self, x):
        # b*hid*h*w
        out = self.dw_bn1(self.dw_conv1(x))
        out = self.dw_act1(out)

        out = self.pw_bn1(self.pw_conv1(out))
        out = self.pw_act1(out)

        # b*hid*h*w
        out = self.dw_bn2(self.dw_conv2(out))
        out = self.dw_ac2(out)

        return out
