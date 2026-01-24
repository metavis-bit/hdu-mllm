from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import normal_


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )

        norm_layer = nn.BatchNorm2d(num_features=out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = nn.Parameter(torch.zeros(out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = nn.Parameter(torch.zeros(out_channels))

        self.conv.add_module("bn", norm_layer)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FFN1D(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(num_features=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # 1D Large Kernel Conv on patches
        feat = patch_tokens.transpose(1, 2)  # (B, C, N-1)
        feat = self.bn(self.conv(feat))
        patch_tokens = feat.transpose(1, 2)  # (B, N-1, C)

        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

