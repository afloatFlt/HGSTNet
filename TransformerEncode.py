from torch import nn
import torch
from torch.nn import functional as F
from einops import rearrange as rearrange
import numbers
import matplotlib.pyplot as plt


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention_C_M(nn.Module):
    def __init__(self, dim, num_heads, bias,LayerNorm_type):
        super(Attention_C_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_1 = self.norm1(x)
        g = self.gate(x_1)

        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)
        attn = F.relu(attn)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out * g
        out = self.project_out(out)
        #out = x+ out
        return out


class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66,LayerNorm_type='WithBias'):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        #self.norm = nn.InstanceNorm2d(num_features=dim, track_running_stats=False)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch*2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch*2, out_channels=num_ch*2, kernel_size=3, stride=1, padding=1, groups=num_ch*2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        #out = out + x
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=128, head=4, expansion_factor=2.66):
        super().__init__()

        self.attn = Attention_C_M(dim=in_ch, num_heads=head,bias=False,LayerNorm_type='WithBias')
        self.feed_forward = FeedForward(dim=in_ch, expansion_factor=expansion_factor,LayerNorm_type='WithBias')

    def forward(self, x):
        x = self.attn(x) + x
        x = self.feed_forward(x) + x
        return x