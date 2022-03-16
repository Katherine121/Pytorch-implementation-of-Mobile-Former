import torch
from torch import nn


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.3):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.scale = channel ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Linear(inner_dim, channel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z):
        m = z.shape[1]
        b = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # b,c,h*w -> b,1,h*w,c
        q = x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        # b,m,d -> b,m,head*c -> b,head,m,c
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        # b,1,h*w,c @ b,head,c,m -> b,head,h*w,m
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        # b,head,h*w,m @ b,head,m,c -> b,head,h*w,c
        out = attn @ v
        # b,head,h*w,c -> b,h*w,head*c
        out = out.transpose(1,2)
        out = out.reshape(b,h*w,-1)
        # b,h*w,head*c -> b,h*w,c
        out = self.to_out(out)
        out = self.drop(out)
        out = out.view(b, c, h, w)
        return x + out
