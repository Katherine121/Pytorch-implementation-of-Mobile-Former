import torch
from torch import nn
import numpy as np


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, z):
        return self.fn(self.norm(z))


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, z):
        res = 0.5 * z * torch.add(1.0, torch.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z * z * z)))
        return res


# inputs: b m d
# output: b m d
class FeedForward(nn.Module):
    def __init__(self, dim=192, mlp_dim=384, dropout=0.3):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            # 高斯误差线性单元激活函数，常用于BERT
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, z):
        return self.net(z)


# inputs: b m d
# output: b m d
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

    def forward(self, z):
        # batch, num, dimension; head
        b = z.shape[0]
        m = z.shape[1]
        # 先经过全连接层获得qkv，然后分割
        qkv = self.to_qkv(z)
        # b,6,192 -> b,6,64 + b,6,64 + b,6,64
        qkv = torch.chunk(qkv, 3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (head d) -> b head n d', h=h),
        #               qkv)  # q: b,6,64 -> b,6,64,1 -> b,6,2,32 -> b,2,6,32 ,2个head，每个head维度32
        q = qkv[0].view(b, m, self.heads, -1)
        q = q.transpose(1, 2)
        k = qkv[1].view(b, m, self.heads, -1)
        k = k.transpose(1, 2)
        v = qkv[2].view(b, m, self.heads, -1)
        v = v.transpose(1, 2)
        # dots = einsum('b head m d, b head m d -> b head m m', q, k) * self.scale  # b,2,6,32 @ b,2,6,32 -> b,2,6,6
        dots = q @ k.transpose(2,3) * self.scale
        attn = self.attend(dots)

        # 每个token经过每个head的attention后的输出
        # out = einsum('b head m m, b head m d -> b head m d', attn, v)  # atten@v b,2,6,6 @ b,2,6,32 -> b,2,6,32
        out = attn @ v

        # out = rearrange(out, 'b head n d -> b n (head d)')  # 合并所有head的输出b,6,64
        out = out.transpose(1,2)
        out = out.reshape(b, m, -1)
        out = self.to_out(out)
        return out


# inputs: b m d
# output: b m d
class Former(nn.Module):
    def __init__(self, dim, heads=2, dim_head=32, dropout=0.3):
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
