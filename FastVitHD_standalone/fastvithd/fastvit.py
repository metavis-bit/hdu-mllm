from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import normal_

from .blocks import basic_blocks
from .mobileone import MobileOneBlock
from .stem import PatchEmbed, convolutional_stem
from .replk import ReparamLargeKernelConv


class GlobalPool2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def pool(self, x) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        x = torch.mean(x, dim=dims, keepdim=False)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        assert x.dim() == 4
        x = self.pool(x)
        x = x @ self.proj
        return x


class FastViT(nn.Module):
    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        stem_scale_branch=True,
        use_feature_fusion: bool = False,
        use_stage5_1d: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_feature_fusion = use_feature_fusion
        self.use_stage5_1d = use_stage5_1d
        if len(layers) == 4:
            self.out_indices = [0, 2, 4, 7]
        elif len(layers) == 5:
            self.out_indices = [0, 2, 4, 7, 10]
        else:
            raise NotImplementedError("FPN is not implemented for more than 5 stages.")

        if self.use_feature_fusion:
            self.fusion_projs = nn.ModuleList()
            for i in range(len(layers)):
                self.fusion_projs.append(
                    ReparamLargeKernelConv(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[-1],
                        kernel_size=7,
                        stride=1,
                        groups=embed_dims[i],
                        small_kernel=3,
                        inference_mode=inference_mode,
                        use_se=True,
                    )
                )

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        self.patch_embed = convolutional_stem(
            3, embed_dims[0], inference_mode, use_scale_branch=stem_scale_branch
        )

        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode))
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
                use_1d=self.use_stage5_1d and i == len(layers) - 1,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        use_se=se_downsamples[i + 1],
                    )
                )
        self.network = nn.ModuleList(network)

        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = (
            nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return_attn = kwargs.get("return_attn", False)
        if self.use_feature_fusion:
            # Calculate target size (final stage resolution)
            target_size = (x.shape[-2] // 16, x.shape[-1] // 16)
            fused_x = 0
            stage_idx = 0
            cls_token_out = None
            last_attn_map = None

            for idx, block in enumerate(self.network):
                # Before Stage 5 starts, if it's 1D, we need to keep track of spatial dims
                if self.use_stage5_1d and idx == self.out_indices[-1]:
                    B, C, H, W = x.shape
                
                # If it's the last stage and we want attn map
                if return_attn and idx == self.out_indices[-1]:
                    # Iterate through the Sequential block
                    for sub_idx, sub_block in enumerate(block):
                        if sub_idx == len(block) - 1:
                            x, last_attn_map = sub_block(x, return_attn=True)
                        else:
                            x = sub_block(x)
                else:
                    x = block(x)

                if idx in self.out_indices:
                    feat = x
                    # If Stage 5 is 1D, we need to extract patches and reshape
                    if self.use_stage5_1d and idx == self.out_indices[-1]:
                        cls_token_out = feat[:, 0]
                        feat = feat[:, 1:]
                        feat = feat.transpose(1, 2).reshape(B, C, H, W)
                    
                    # Pool early to save memory
                    feat = F.adaptive_avg_pool2d(feat, target_size)
                    # Project and accumulate
                    fused_x = fused_x + self.fusion_projs[stage_idx](feat)
                    stage_idx += 1
            
            # Apply simple averaging (mean fusion)
            if stage_idx > 0:
                fused_x = fused_x / stage_idx
            
            if self.use_stage5_1d:
                if return_attn:
                    # Extract cls attention map (B, num_heads, 1, N)
                    # Exclude cls-to-cls self attention
                    cls_attn = last_attn_map[:, :, 0:1, 1:]
                    return cls_token_out, fused_x, cls_attn
                return cls_token_out, fused_x
            return fused_x
        else:
            cls_token_out = None
            last_attn_map = None
            for idx, block in enumerate(self.network):
                if self.use_stage5_1d and idx == self.out_indices[-1]:
                    B, C, H, W = x.shape
                
                # If it's the last stage and we want attn map
                if return_attn and idx == self.out_indices[-1]:
                    # Iterate through the Sequential block
                    for sub_idx, sub_block in enumerate(block):
                        if sub_idx == len(block) - 1:
                            x, last_attn_map = sub_block(x, return_attn=True)
                        else:
                            x = sub_block(x)
                else:
                    x = block(x)

                # If we just finished the last block of the last stage and it's 1D
                if self.use_stage5_1d and idx == self.out_indices[-1]:
                    cls_token_out = x[:, 0]
                    patch_tokens = x[:, 1:]
                    x = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            
            if self.use_stage5_1d:
                if return_attn:
                    # Extract cls attention map (B, num_heads, 1, N)
                    # Exclude cls-to-cls self attention
                    cls_attn = last_attn_map[:, :, 0:1, 1:]
                    return cls_token_out, x, cls_attn
                return cls_token_out, x
            return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[Tensor, Dict[str, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        return_attn = kwargs.get("return_attn", False)
        x = self.forward_embeddings(x)
        res = self.forward_tokens(x, return_attn=return_attn)

        cls_attn = None
        if isinstance(res, tuple):
            if len(res) == 3:
                cls_token, x, cls_attn = res
            else:
                cls_token, x = res
        else:
            x = res

        x = self.conv_exp(x)

        if self.num_classes > 0:
            x_pool = self.gap(x)
            x_pool = x_pool.view(x_pool.size(0), -1)
            # If we have cls_token, maybe use it for head? 
            # But the user asked for separate return.
            cls_out = self.head(x_pool)
        else:
            cls_out = self.head(x)

        if kwargs.get("return_image_embeddings", False):
            out = {"logits": cls_out, "image_embeddings": x}
            if self.use_stage5_1d:
                out["cls_token"] = cls_token
            if cls_attn is not None:
                out["cls_attn"] = cls_attn
            return out
        
        if self.use_stage5_1d:
            if cls_attn is not None:
                return cls_token, x, cls_attn
            return cls_token, x
            
        return cls_out

