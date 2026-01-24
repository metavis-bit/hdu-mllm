import torch
import torch.nn as nn
from typing import Tuple, Union


class MHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)
        else:
            B, N, C = shape

        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        attn_out = None
        if return_attn:
            attn_out = attn

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        if return_attn:
            return x, attn_out
        return x

