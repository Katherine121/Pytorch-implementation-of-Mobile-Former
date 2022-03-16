import numpy as np
import torch
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, z):
        return 0.5*z*(1.0+torch.tanh(np.sqrt(2.0/np.pi)*(z+0.044715*z*z*z)))


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.3):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        # 高斯误差线性单元激活函数，常用于BERT
        self.gelu = GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, z):
        z = self.fc1(z)
        z = self.gelu(z)
        z = self.drop1(z)
        z = self.fc2(z)
        z = self.drop2(z)
        return z


class Attention(nn.Module):
    def __init__(self, dim=192, heads=2, dim_head=32, dropout=0.3):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        # batch, num
        b = z.shape[0]
        m = z.shape[1]
        # 先经过全连接层获得qkv，然后分割
        # b,6,192 -> b,6,64 + b,6,64 + b,6,64
        qkv = self.to_qkv(z).chunk(3, dim=-1)
        # b,6,64 -> b,6,2,32 -> b,2,6,32
        q = qkv[0].view(b, m, self.heads, -1)
        q = q.transpose(1, 2)
        k = qkv[1].view(b, m, self.heads, -1)
        k = k.transpose(1, 2)
        v = qkv[2].view(b, m, self.heads, -1)
        v = v.transpose(1, 2)

        # b,2,6,32 @ b,2,32,6 -> b,2,6,6
        dots = q @ k.transpose(2,3) * self.scale
        attn = self.attend(dots)
        # 每个token经过每个head的attention后的输出
        # b,2,6,6 @ b,2,6,32 -> b,2,6,32
        out = attn @ v

        # b,2,6,32 -> b,6,64
        out = out.transpose(1, 2)
        out = out.reshape(b, m, -1)
        # b,6,64 -> b,6,192
        out = self.to_out(out)
        out = self.drop(out)
        return out


# inputs: b m d
# output: b m d
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
