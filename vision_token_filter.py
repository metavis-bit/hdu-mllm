from typing import Optional

import torch


def filter_vision_tokens_with_cls_attn(
    cls_attn: Optional[torch.Tensor],
    text_tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vision_tokens.dim() != 3:
        raise ValueError(f"vision_tokens must be (B, N, D), got {tuple(vision_tokens.shape)}")
    if text_tokens is None or text_tokens.numel() == 0:
        raise ValueError("text_tokens must be a non-empty tensor")

    bsz, num_tokens, dim = vision_tokens.shape
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0:
        keep_ratio = 1.0
    text_tokens = text_tokens.to(dtype=vision_tokens.dtype, device=vision_tokens.device)
    attn_scores = torch.matmul(vision_tokens.float(), text_tokens.float().transpose(1, 2))
    attn_scores = torch.nan_to_num(attn_scores, nan=0.0, posinf=0.0, neginf=0.0)

    attn_sum = attn_scores.sum(dim=-1)
    attn_min = attn_sum.min(dim=1, keepdim=True).values
    attn_max = attn_sum.max(dim=1, keepdim=True).values
    attn_norm = (attn_sum - attn_min) / (attn_max - attn_min + 1e-6)
    attn_norm = torch.nan_to_num(attn_norm, nan=0.0, posinf=0.0, neginf=0.0)

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
    entropy = -(attn_probs * torch.log(attn_probs.clamp_min(1e-8))).sum(dim=-1)
    entropy = entropy / torch.log(torch.tensor(attn_scores.size(-1), device=entropy.device, dtype=entropy.dtype))
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    ent_min = entropy.min(dim=1, keepdim=True).values
    ent_max = entropy.max(dim=1, keepdim=True).values
    ent_norm = (entropy - ent_min) / (ent_max - ent_min + 1e-6)
    ent_norm = torch.nan_to_num(ent_norm, nan=0.0, posinf=0.0, neginf=0.0)

    token_probs = torch.softmax(vision_tokens.float(), dim=-1)
    token_probs = torch.nan_to_num(token_probs, nan=0.0, posinf=0.0, neginf=0.0)
    token_entropy = -(token_probs * torch.log(token_probs.clamp_min(1e-8))).sum(dim=-1)
    token_entropy = token_entropy / torch.log(
        torch.tensor(vision_tokens.size(-1), device=token_entropy.device, dtype=token_entropy.dtype)
    )
    token_entropy = torch.nan_to_num(token_entropy, nan=0.0, posinf=0.0, neginf=0.0)
    token_ent_min = token_entropy.min(dim=1, keepdim=True).values
    token_ent_max = token_entropy.max(dim=1, keepdim=True).values
    token_ent_norm = (token_entropy - token_ent_min) / (token_ent_max - token_ent_min + 1e-6)
    token_ent_norm = torch.nan_to_num(token_ent_norm, nan=0.0, posinf=0.0, neginf=0.0)

    cls_norm = torch.zeros_like(attn_norm)
    if cls_attn is not None and cls_attn.numel() != 0:
        cls_scores = cls_attn
        if cls_scores.dim() >= 3:
            cls_scores = cls_scores.mean(dim=1)
        if cls_scores.dim() >= 3:
            cls_scores = cls_scores.mean(dim=-2)
        if cls_scores.dim() != 2:
            cls_scores = cls_scores.view(bsz, -1)
        if cls_scores.shape[1] > num_tokens:
            cls_scores = cls_scores[:, :num_tokens]
        elif cls_scores.shape[1] < num_tokens:
            pad = torch.zeros((bsz, num_tokens - cls_scores.shape[1]), device=cls_scores.device, dtype=cls_scores.dtype)
            cls_scores = torch.cat([cls_scores, pad], dim=1)
        cls_min = cls_scores.min(dim=1, keepdim=True).values
        cls_max = cls_scores.max(dim=1, keepdim=True).values
        cls_norm = (cls_scores - cls_min) / (cls_max - cls_min + 1e-6)
        cls_norm = torch.nan_to_num(cls_norm, nan=0.0, posinf=0.0, neginf=0.0)

    attn_weight = 0.6
    ent_weight = 0.2
    token_ent_weight = 0.2
    vision_token_scores = attn_weight * attn_norm + ent_weight * (1.0 - ent_norm) + token_ent_weight * (1.0 - token_ent_norm)
    # Anchor-Former style prior: bias scores using cls attention.
    vision_token_scores = vision_token_scores * (0.5 + 0.5 * cls_norm)
    vision_token_scores = torch.nan_to_num(vision_token_scores, nan=0.0, posinf=0.0, neginf=0.0)

    keep_count = max(1, min(num_tokens, int(num_tokens * keep_ratio)))
    topk = torch.topk(vision_token_scores, k=keep_count, dim=1)
    keep_idx = torch.sort(topk.indices, dim=1).values
    batch_idx = torch.arange(vision_tokens.size(0), device=vision_tokens.device).unsqueeze(1)
    filtered_vision_tokens = vision_tokens[batch_idx, keep_idx]

    all_idx = torch.arange(vision_tokens.size(1), device=vision_tokens.device).unsqueeze(0)
    keep_mask = torch.zeros_like(all_idx, dtype=torch.bool).repeat(vision_tokens.size(0), 1)
    keep_mask.scatter_(1, keep_idx, True)
    drop_idx = all_idx.repeat(vision_tokens.size(0), 1)[~keep_mask].view(vision_tokens.size(0), -1)
    dropped_tokens = vision_tokens[batch_idx, drop_idx]
    dropped_scores = vision_token_scores[batch_idx, drop_idx]

    knn_k = 5
    syn_tokens_list = []
    for b in range(vision_tokens.size(0)):
        dt = dropped_tokens[b]
        if dt.numel() == 0:
            syn_tokens_list.append(None)
            continue
        nd = dt.size(0)
        k = min(knn_k, nd)
        anchor_scores = torch.nan_to_num(dropped_scores[b].float(), nan=0.0, posinf=0.0, neginf=0.0)
        score_mean = anchor_scores.abs().mean()
        score_std = anchor_scores.std()
        if not torch.isfinite(score_mean) or not torch.isfinite(score_std):
            adaptive_ratio = 0.2
        else:
            adaptive_ratio = torch.clamp(score_std / (score_mean + 1e-6), 0.1, 0.3).item()
        if not (adaptive_ratio == adaptive_ratio):
            adaptive_ratio = 0.2
        synth_count = max(1, int(nd * adaptive_ratio))
        anchor_idx = torch.topk(anchor_scores, k=synth_count).indices
        dt_norm = dt / (dt.norm(dim=-1, keepdim=True) + 1e-6)
        dt_norm = torch.nan_to_num(dt_norm, nan=0.0, posinf=0.0, neginf=0.0)
        sims = dt_norm @ dt_norm.transpose(0, 1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        synth_tokens = []
        for ai in anchor_idx:
            knn_idx = torch.topk(sims[ai], k=k).indices
            synth_tokens.append(dt[knn_idx].mean(dim=0))
        syn_tokens_list.append(torch.stack(synth_tokens, dim=0))

    filtered_list = []
    lengths: list[int] = []
    for b in range(filtered_vision_tokens.size(0)):
        f = filtered_vision_tokens[b]
        st = syn_tokens_list[b]
        if st is not None:
            f = torch.cat([f, st], dim=0)
        filtered_list.append(f)
        lengths.append(int(f.shape[0]))

    max_len = max(lengths) if lengths else 0
    out_tokens = torch.zeros((bsz, max_len, dim), device=vision_tokens.device, dtype=vision_tokens.dtype)
    out_mask = torch.zeros((bsz, max_len), device=vision_tokens.device, dtype=torch.int8)
    for b, toks in enumerate(filtered_list):
        L = toks.shape[0]
        if L > 0:
            out_tokens[b, :L] = toks
            out_mask[b, :L] = 1
    return out_tokens, out_mask
