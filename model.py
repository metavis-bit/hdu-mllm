import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attention import flash_attention_fn
import math
from typing import Optional, Tuple

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

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)

class StaggeredVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int, base_width: int = 64):
        super().__init__()
        # 224x224 -> 112x112
        self.stage1_down = nn.Conv2d(3, base_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage1_res = ConvResidualBlock(base_width, base_width * 2)
        
        # 112x112 -> 56x56
        self.stage2_down = nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2_res = ConvResidualBlock(base_width * 4, base_width * 4)
        
        # 56x56 -> 28x28
        self.stage3_down = nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3_res = ConvResidualBlock(base_width * 8, base_width * 8)
        
        # 28x28 -> 14x14
        self.stage4_down = nn.Conv2d(base_width * 8, base_width * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage4_res = ConvResidualBlock(base_width * 16, base_width * 16)

        # 14x14 -> 7x7
        self.stage5_down = nn.Conv2d(base_width * 16, base_width * 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage5_res = ConvResidualBlock(base_width * 32, base_width * 32)
        
        # Final projection: (B, base_width*32, 7, 7) -> (B, 49, hidden_size)
        self.proj = nn.Linear(base_width * 32, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Step-by-step downsampling
        x = self.stage1_down(images)
        x = self.stage1_res(x)
        
        x = self.stage2_down(x)
        x = self.stage2_res(x)
        
        x = self.stage3_down(x)
        x = self.stage3_res(x)
        
        x = self.stage4_down(x)
        x = self.stage4_res(x)

        x = self.stage5_down(x)
        x = self.stage5_res(x)
        
        # Flatten and project
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.proj(x)

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

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config['hidden_size'], config['intermediate_size'])
        self.input_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, x, cos, sin, attention_mask=None):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask=attention_mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

    def generate_step(self, x, cos, sin, past_key_value=None, use_cache=False):
        attn_out, present_key_value = self.self_attn.generate_step(
            self.input_layernorm(x), 
            cos, 
            sin, 
            past_key_value=past_key_value, 
            use_cache=use_cache
        )
        h = x + attn_out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out, present_key_value

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.vision = StaggeredVisionEncoder(config['hidden_size'], base_width=64)
        
        cos, sin = precompute_freqs_cis(config['head_dim'], config['max_position_embeddings'], config['rope_theta'])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
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
            x = layer(x, cos, sin, attention_mask=attention_mask)
        
        return self.norm(x)

    def generate_step(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False):
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
            x, present_kv = layer.generate_step(x, cos, sin, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present_kv)
        
        return self.norm(x), new_past_key_values

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        if config.get('tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """
        Standard training forward pass.
        """
        hidden_states = self.model(input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states)
        return logits

    def generate_step(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False):
        """
        Inference step with KV Cache support.
        """
        hidden_states, next_past_key_values = self.model.generate_step(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        return logits, next_past_key_values

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.vision(images)
