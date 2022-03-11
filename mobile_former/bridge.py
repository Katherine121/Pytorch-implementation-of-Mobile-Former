import torch
from torch import nn


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.3):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        m = z.size(1)
        b = x.size(0)
        c = x.size(1)
        h = x.size(2)
        w = x.size(3)
        # former作为Q
        # b, m, d -> b, m, head*c -> b,head,m,c
        q = self.to_q(z).view(b, self.heads, m, c)
        # mobile作为K, V
        # b, c, h, w -> b, 1, h*w, c
        x = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = x
        # for i in range(b - 1):
        #     k = torch.cat((k, x), dim=1)
        # 矩阵相乘 b, head, m, c @ b, 1, c, h*w -> b, head, m, h*w
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        # 矩阵相乘 b, head, m, h*w @ b, 1, h*w, c -> b, head, m, c
        out = attn @ k
        # b, head, m, c ->b,m,head*c
        out = out.transpose(1, 2)
        out = out.reshape(b, m, self.heads * c)
        # b,m,head*c -> b,m,d
        out = self.to_out(out)
        out = torch.add(out, z)
        return out


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.3):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        m = z.shape[1]
        b = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # b,c,h*w -> b,1,h*w,c
        x1 = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        q = x1
        # for i in range(b - 1):
        #     q = torch.cat((q, x1), dim=1)
        # b,m,d -> b,m,head*c -> b,head,m,c
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        # b,1,h*w,c @ b,head,c,m -> b,head,h*w,m
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        # b,head,h*w,m @ b,head,m,c -> b,head,h*w,c
        out = attn @ v
        # b,head,c,h*w -> b,h*w,head*c
        out = out.transpose(1, 2)
        res = out.reshape(b, h * w, self.heads * c)
        # b,h*w,head*c -> b,h*w,c
        out = self.to_out(res)
        out = out.view(b, c, h, w)
        out = torch.add(out, x)
        return out
