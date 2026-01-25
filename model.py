import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attention import flash_attention_fn
import math
from typing import Optional, Tuple
from FastVitHD_standalone import fastvithd

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FastViTHDVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = fastvithd(
            use_feature_fusion=True,
            use_stage5_1d=True,
            inference_mode=False,
        )
        self.out_dim = int(getattr(self.backbone.conv_exp, "out_channels"))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        res = self.backbone(images, return_attn=True)
        x = res
        cls_attn = None
        if isinstance(res, tuple):
            if len(res) == 3:
                _, x, cls_attn = res
            elif len(res) == 2:
                _, x = res
            else:
                x = res[-1]

        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.dim() != 3:
            raise ValueError(f"Unexpected FastViTHD output shape: {tuple(x.shape)}")

        return x, cls_attn


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = min(out_dim * 4, 8192)
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    # cos, sin: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim)
    # xq, xk: (bsz, seq_len, n_heads, head_dim)
    cos = torch.cat([cos, cos], dim=-1).view(1, cos.shape[0], 1, -1)
    sin = torch.cat([sin, sin], dim=-1).view(1, sin.shape[0], 1, -1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = config['head_dim']
        self.num_kv_heads = config['num_key_value_heads']
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Qwen3 often uses QK-Norm per head
        self.q_norm = RMSNorm(self.head_dim, eps=config['rms_norm_eps'])
        self.k_norm = RMSNorm(self.head_dim, eps=config['rms_norm_eps'])

    def forward(self, x, cos, sin, attention_mask=None):
        """
        Standard training forward pass (full computation).
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # GQA
        xk = torch.repeat_interleave(xk, self.num_kv_groups, dim=2)
        xv = torch.repeat_interleave(xv, self.num_kv_groups, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Training usually uses Flash Attention for full sequence
        output = flash_attention_fn(xq, xk, xv, causal=True, key_padding_mask=attention_mask)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

    def generate_step(self, x, cos, sin, past_key_value=None, use_cache=False):
        """
        Inference step with KV Cache support.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Apply QK-Norm per head
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            # past_key_value: (k, v) where k,v are (bsz, prev_seqlen, num_kv_heads, head_dim)
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        present_key_value = (xk, xv) if use_cache else None

        # GQA: Repeat KV heads
        xk_repeated = torch.repeat_interleave(xk, self.num_kv_groups, dim=2)
        xv_repeated = torch.repeat_interleave(xv, self.num_kv_groups, dim=2)

        # Use our custom Flash Attention operator
        # Input shape expected: (B, H, N, D)
        xq = xq.transpose(1, 2)
        xk_repeated = xk_repeated.transpose(1, 2)
        xv_repeated = xv_repeated.transpose(1, 2)
        
        if seqlen > 1 and past_key_value is None:
            # First Prefill stage: Use custom Flash Attention (Q.len == K.len)
            output = flash_attention_fn(xq, xk_repeated, xv_repeated, causal=True)
        else:
            # Incremental Prefill or Decoding stage: Use ordinary attention
            # Ordinary attention handles seqlen_q != seqlen_k naturally
            scores = torch.matmul(xq, xk_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)
            if seqlen > 1:
                # generate rectangular causal mask
                total_len = xk_repeated.shape[2]
                mask = torch.full((seqlen, total_len), float("-inf"), device=xq.device)
                mask = torch.triu(mask, diagonal=total_len - seqlen + 1)
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv_repeated)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output), present_key_value


class MultiHeadCrossAttention(nn.Module):
    """
    DeepSeek-style MHC connector: language tokens attend to visual tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Match the self-attention's per-head normalization.
        self.q_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])

    def forward(
        self,
        x: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        vlen = vision_tokens.shape[1]

        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(vision_tokens).view(bsz, vlen, self.num_heads, self.head_dim)
        v = self.v_proj(vision_tokens).view(bsz, vlen, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, V, D)
        v = v.transpose(1, 2)  # (B, H, V, D)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        has_vision = None
        if vision_attention_mask is not None and vision_attention_mask.numel() != 0:
            # vision_attention_mask: (B, V), 1=valid, 0=pad
            valid_counts = vision_attention_mask.sum(dim=-1)
            has_vision = valid_counts > 0
            safe_mask = vision_attention_mask.to(dtype=torch.bool)
            if (~has_vision).any() and safe_mask.shape[1] > 0:
                # Prevent all -inf rows, then zero them out after attention.
                safe_mask = safe_mask.clone()
                safe_mask[~has_vision, 0] = True
            mask = (~safe_mask)[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attn, v)
        if has_vision is not None:
            output = output * has_vision[:, None, None].to(output.dtype)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.use_mhc = bool(config.get("use_mhc", True))
        self.mhc = MultiHeadCrossAttention(config) if self.use_mhc else None
        gate_init = float(config.get("mhc_gate_init", 0.0))
        self.mhc_gate = nn.Parameter(torch.tensor(gate_init)) if self.use_mhc else None
        self.mhc_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]) if self.use_mhc else None
        self.mlp = Qwen3MLP(config['hidden_size'], config['intermediate_size'])
        self.input_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, x, cos, sin, attention_mask=None, vision_tokens=None, vision_attention_mask=None):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask=attention_mask)
        if self.use_mhc and vision_tokens is not None and vision_tokens.numel() != 0:
            mhc_out = self.mhc(self.mhc_norm(h), vision_tokens, vision_attention_mask=vision_attention_mask)
            h = h + self.mhc_gate * mhc_out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

    def generate_step(
        self,
        x,
        cos,
        sin,
        past_key_value=None,
        use_cache=False,
        vision_tokens=None,
        vision_attention_mask=None,
    ):
        attn_out, present_key_value = self.self_attn.generate_step(
            self.input_layernorm(x), 
            cos, 
            sin, 
            past_key_value=past_key_value, 
            use_cache=use_cache
        )
        h = x + attn_out
        if self.use_mhc and vision_tokens is not None and vision_tokens.numel() != 0:
            mhc_out = self.mhc(self.mhc_norm(h), vision_tokens, vision_attention_mask=vision_attention_mask)
            h = h + self.mhc_gate * mhc_out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out, present_key_value

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.vision_backbone = FastViTHDVisionEncoder()
        self.vision_projector = VisionProjector(self.vision_backbone.out_dim, config['hidden_size'])
        
        cos, sin = precompute_freqs_cis(config['head_dim'], config['max_position_embeddings'], config['rope_theta'])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Standard training forward pass.
        """
        if inputs_embeds is None:
            bsz, seqlen = input_ids.shape
            x = self.embed_tokens(input_ids)
        else:
            bsz, seqlen, _ = inputs_embeds.shape
            x = inputs_embeds
        
        cos = self.cos[0:seqlen].to(x.device)
        sin = self.sin[0:seqlen].to(x.device)
        
        for layer in self.layers:
            x = layer(
                x,
                cos,
                sin,
                attention_mask=attention_mask,
                vision_tokens=vision_tokens,
                vision_attention_mask=vision_attention_mask,
            )
        
        return self.norm(x)

    def generate_step(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Inference step with KV Cache support.
        """
        if inputs_embeds is None:
            bsz, seqlen = input_ids.shape
            x = self.embed_tokens(input_ids)
        else:
            bsz, seqlen, _ = inputs_embeds.shape
            x = inputs_embeds
        
        # Determine current position for RoPE
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]
        
        # Slice cos/sin for current segment
        cos = self.cos[past_length : past_length + seqlen].to(x.device)
        sin = self.sin[past_length : past_length + seqlen].to(x.device)
        
        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer.generate_step(
                x,
                cos,
                sin,
                past_key_value=past_kv,
                use_cache=use_cache,
                vision_tokens=vision_tokens,
                vision_attention_mask=vision_attention_mask,
            )
            if use_cache:
                new_past_key_values.append(present_kv)
        
        return self.norm(x), new_past_key_values

    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens, cls_attn = self.vision_backbone(images)
        return self.vision_projector(tokens), cls_attn

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        if config.get('tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Standard training forward pass.
        """
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            vision_tokens=vision_tokens,
            vision_attention_mask=vision_attention_mask,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def generate_step(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Inference step with KV Cache support.
        """
        hidden_states, next_past_key_values = self.model.generate_step(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, 
            use_cache=use_cache,
            vision_tokens=vision_tokens,
            vision_attention_mask=vision_attention_mask,
        )
        logits = self.lm_head(hidden_states)
        return logits, next_past_key_values

    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.model.encode_images(images)
