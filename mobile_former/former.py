import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1.0+F.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x*x*x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # 高斯误差线性单元激活函数，常用于BERT
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=192, heads=2, dim_head=32, dropout=0.3):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        # 如果不是多头，就没必要合并了
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 如果不是多头，就没必要合并了
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, z):  # b,6,192
        # batch, num, dimension; head
        b = z.shape[0]
        m = z.shape[1]
        # 先经过全连接层获得qkv，然后分割
        qkv = self.to_qkv(z).chunk(3, dim=-1)  # b,6,192 -> b,6,64 + b,6,64 + b,6,64
        q = qkv[0].view(b, m, self.heads, -1)
        q = q.transpose(1, 2)
        k = qkv[1].view(b, m, self.heads, -1)
        k = k.transpose(1, 2)
        v = qkv[2].view(b, m, self.heads, -1)
        v = v.transpose(1, 2)

        dots = q @ k.transpose(2,3) * self.scale
        attn = self.attend(dots)
        # 每个token经过每个head的attention后的输出
        out = attn @ v

        out = out.transpose(1, 2)
        out = out.reshape(b, m, -1)
        return self.to_out(out)


# inputs: n m d
# output: n m d
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        # 2 instead of 4
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])

        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, z):
        attn = self.layers[0][0]
        ff = self.layers[0][1]
        # 残差连接
        z = attn(z) + z
        z = ff(z) + z
        return z
