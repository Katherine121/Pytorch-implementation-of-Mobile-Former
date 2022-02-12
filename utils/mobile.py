import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import MyDyRelu
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

# 没用到
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

# 没用到
class SeModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        b, c, _, _ = se.size()
        se = se.view(b, c)
        se = self.se(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        # [1.0000, 1.0000, 0.5000, 0.5000]
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        # [1, 0, 0, 0]
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = MyDyRelu(2)

        self.conv2 = nn.Conv2d(hid, hid, kernel_size=ks, stride=stride,
                               padding=ks // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.act2 = MyDyRelu(2)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        # 取第一个token
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k*hid
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k*hid
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*hid -> b hid 2*k                                  2*k            2*k
        # 前k个是第一个超参，后k个是第二个超参
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        # b*hid*h*w, b*hid*2k
        out = self.bn1(self.conv1(x))
        out_ = [out, relu_coefs]
        out = self.act1(out_)

        # b*hid*h*w, b*hid*2k
        out = self.bn2(self.conv2(out))
        out_ = [out, relu_coefs]
        out = self.act2(out_)

        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        # 如果图片没有下采样，则残差连接，此模块没有下采样所以要残差连接
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid, self.out = hid, out
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()
        # [1.0000, 1.0000, 0.5000, 0.5000]
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        # [1, 0, 0, 0]
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.dw_conv1 = nn.Conv2d(inp, hid, kernel_size=ks, stride=stride,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_act1 = MyDyRelu(2)

        self.pw_conv1 = nn.Conv2d(hid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(inp, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_act2 = MyDyRelu(2)

        self.pw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        # 取第一个token
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k*hid
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k*hid
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # 第一个参数*1，左边+1，右边+0；第二个参数*0.5，+0
        # b 2*k*hid -> b hid 2*k                                  2*k            2*k
        # 前k个是第一个超参，后k个是第二个超参
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        # b*hid*h*w, b*hid*2k
        out = self.dw_bn1(self.dw_conv1(x))
        out_ = [out, relu_coefs]
        out = self.dw_act1(out_)

        # 这里用的是普通relu，所以不加第一个token
        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        # b*hid*h*w, b*hid*2k
        out = self.dw_bn2(self.dw_conv2(out))
        out_ = [out, relu_coefs]
        out = self.dw_act2(out_)

        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        # 如果图片没有下采样，则残差连接，此模块有下采样所以不残差连接
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
