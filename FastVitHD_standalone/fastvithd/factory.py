from __future__ import annotations

from functools import partial

from .cfg import default_cfgs
from .cpe import RepCPE
from .fastvit import FastViT
from .norm import LayerNormChannel


def fastvithd(use_feature_fusion: bool = False, use_stage5_1d: bool = False, inference_mode: bool = False, **kwargs):
    layers = [2, 12, 24, 4, 2]
    embed_dims = [96, 192, 384, 768, 1536]
    mlp_ratios = [4, 4, 4, 4, 4]
    downsamples = [True, True, True, True, True]
    pos_embs = [
        None,
        None,
        None,
        partial(RepCPE, spatial_shape=(7, 7)),
        partial(RepCPE, spatial_shape=(7, 7)),
    ]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        norm_layer=LayerNormChannel,
        stem_scale_branch=False,
        inference_mode=inference_mode,
        num_classes=0,
        use_feature_fusion=use_feature_fusion,
        use_stage5_1d=use_stage5_1d,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

