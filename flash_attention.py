import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1. Flash Attention Forward Kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BQ': 64, 'BK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 128, 'BK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 64, 'BK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 128, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BQ': 32, 'BK': 64}, num_warps=2, num_stages=2),
    ],
    key=['n'],
)
@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, O, L,  # 输入输出张量
    M,  # key_padding_mask, shape: (B*H, n), 1=valid, 0=pad
    stride_qb, stride_qn, stride_qd,  # Q 张量的步长
    stride_kb, stride_kn, stride_kd,  # K 张量的步长
    stride_vb, stride_vn, stride_vd,  # V 张量的步长
    stride_ob, stride_on, stride_od,  # O 张量的步长
    stride_lb, stride_ln,             # L 张量的步长 (用于存储 softmax 的 logsumexp)
    stride_mb, stride_mn,             # M 张量的步长
    n,                                # 序列长度
    d_scale,                          # 缩放因子 1/√d
    IS_CAUSAL: tl.constexpr,          # 是否使用因果掩码
    HAS_MASK: tl.constexpr,           # 是否使用 Padding Mask
    BQ: tl.constexpr,                 # Q 块大小
    BK: tl.constexpr,                 # K 块大小
    D: tl.constexpr,                  # 特征维度
    eps: tl.constexpr = 1e-6,         # 数值稳定性常数
):
    # 获取当前线程块的 ID
    pid_b = tl.program_id(0)   # batch * num_heads 维度
    pid_tq = tl.program_id(1)  # Q 块维度
    
    # 创建块指针 - Triton 的高级内存访问抽象
    q_block_ptr = tl.make_block_ptr(
        base=Q + pid_b * stride_qb,
        shape=(n, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_tq * BQ, 0),
        block_shape=(BQ, D),
        order=(1, 0),
    )
    
    k_block_ptr = tl.make_block_ptr(
        base=K + pid_b * stride_kb,
        shape=(D, n),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BK),
        order=(0, 1),
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=V + pid_b * stride_vb,
        shape=(n, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    
    o_block_ptr = tl.make_block_ptr(
        base=O + pid_b * stride_ob,
        shape=(n, D),
        strides=(stride_on, stride_od),
        offsets=(pid_tq * BQ, 0),
        block_shape=(BQ, D),
        order=(1, 0),
    )
    
    l_ptrs = L + pid_b * stride_lb + (pid_tq * BQ + tl.arange(0, BQ)) * stride_ln
    
    offs_q = pid_tq * BQ + tl.arange(0, BQ)
    if HAS_MASK:
        q_valid = tl.load(
            M + pid_b * stride_mb + offs_q * stride_mn,
            mask=offs_q < n,
            other=0,
        ).to(tl.int1)
    else:
        q_valid = tl.full([BQ], value=1, dtype=tl.int1)

    # 初始化累加器
    m_i = tl.where(q_valid, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.where(q_valid, 0.0, 1.0).to(tl.float32)
    o_i = tl.zeros([BQ, D], dtype=tl.float32)
    
    # 加载 Q 块并缩放
    q_i = tl.load(q_block_ptr, boundary_check=(0, 1))
    q_dtype = q_i.dtype
    q_i = (q_i * d_scale).to(q_dtype)
    
    # 计算循环边界（支持因果掩码）
    loop_end = tl.cdiv(n, BK)
    if IS_CAUSAL:
        loop_end = tl.cdiv((pid_tq + 1) * BQ, BK)
    
    # 主循环：遍历所有 K, V 块
    for j in range(loop_end):
        # 加载当前 K, V 块
        k_j = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1))
        
        # 计算注意力分数：S = Q @ K^T
        s_ij = tl.dot(q_i, k_j)

        if HAS_MASK:
            offs_k = j * BK + tl.arange(0, BK)
            k_valid = tl.load(
                M + pid_b * stride_mb + offs_k * stride_mn,
                mask=offs_k < n,
                other=0,
            ).to(tl.int1)
            s_ij += tl.where(k_valid[None, :], 0.0, float("-inf"))

        # 应用因果掩码
        if IS_CAUSAL:
            offs_k = j * BK + tl.arange(0, BK)
            s_ij += tl.where(offs_q[:, None] >= offs_k[None, :], 0, float("-inf"))

        # 在线 Softmax 更新
        m_next_raw = tl.maximum(m_i, tl.max(s_ij, axis=1))
        m_next = tl.where(q_valid, m_next_raw, m_i)
        p_ij = tl.exp(s_ij - m_next[:, None])
        p_ij = tl.where(q_valid[:, None], p_ij, 0.0)

        scale = tl.exp(m_i - m_next)
        scale = tl.where(q_valid, scale, 1.0)
        l_i = tl.where(q_valid, scale * l_i + tl.sum(p_ij, axis=1), l_i)

        o_i = tl.where(q_valid[:, None], scale[:, None] * o_i + tl.dot(p_ij.to(v_j.dtype), v_j), o_i)
        
        # 更新状态
        m_i = m_next
        
        # 移动到下一个块
        k_block_ptr = tl.advance(k_block_ptr, (0, BK))
        v_block_ptr = tl.advance(v_block_ptr, (BK, 0))
    
    # 最终归一化
    o_i = tl.where(q_valid[:, None], o_i / l_i[:, None], 0.0)
    
    # 存储结果
    tl.store(o_block_ptr, o_i.to(q_dtype), boundary_check=(0, 1))
    # 存储 L (LogSumExp) 用于反向传播
    l_i = tl.where(q_valid, m_i + tl.log(l_i + eps), 0.0)
    tl.store(l_ptrs, l_i, mask=offs_q < n)

# -----------------------------------------------------------------------------
# 2. Flash Attention Backward Kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BQ': 64, 'BK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 32, 'BK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 128, 'BK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 64, 'BK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BQ': 32, 'BK': 64}, num_warps=4, num_stages=2),
    ],
    key=['n'],
)
@triton.jit
def _flash_attention_bwd_kernel(
    Q, K, V, O, L,
    M,
    DO, DQ, DK, DV,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    stride_mb, stride_mn,
    n,
    d_scale,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_tk = tl.program_id(1) # 以 K 为基准遍历 Q
    
    # 初始化 dK 和 dV 的累加器
    dk = tl.zeros([BK, D], dtype=tl.float32)
    dv = tl.zeros([BK, D], dtype=tl.float32)
    
    # 加载 K 和 V 块
    k_block_ptr = tl.make_block_ptr(
        base=K + pid_b * stride_kb,
        shape=(n, D),
        strides=(stride_kn, stride_kd),
        offsets=(pid_tk * BK, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + pid_b * stride_vb,
        shape=(n, D),
        strides=(stride_vn, stride_vd),
        offsets=(pid_tk * BK, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    
    k_j = tl.load(k_block_ptr, boundary_check=(0, 1))
    v_j = tl.load(v_block_ptr, boundary_check=(0, 1))
    k_dtype = k_j.dtype
    
    # 转置 K 块用于计算 Q @ K^T
    kt_j = tl.trans(k_j)
    
    # 遍历 Q 块
    loop_start = 0
    if IS_CAUSAL:
        loop_start = (pid_tk * BK) // BQ
        
    for i in range(loop_start, tl.cdiv(n, BQ)):
        # 加载 Q, O, DO, L 块
        q_block_ptr = tl.make_block_ptr(
            base=Q + pid_b * stride_qb,
            shape=(n, D),
            strides=(stride_qn, stride_qd),
            offsets=(i * BQ, 0),
            block_shape=(BQ, D),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            base=O + pid_b * stride_ob,
            shape=(n, D),
            strides=(stride_on, stride_od),
            offsets=(i * BQ, 0),
            block_shape=(BQ, D),
            order=(1, 0),
        )
        do_block_ptr = tl.make_block_ptr(
            base=DO + pid_b * stride_ob,
            shape=(n, D),
            strides=(stride_on, stride_od),
            offsets=(i * BQ, 0),
            block_shape=(BQ, D),
            order=(1, 0),
        )
        
        q_i = tl.load(q_block_ptr, boundary_check=(0, 1))
        o_i = tl.load(o_block_ptr, boundary_check=(0, 1))
        do_i = tl.load(do_block_ptr, boundary_check=(0, 1))
        l_i = tl.load(L + pid_b * stride_lb + (i * BQ + tl.arange(0, BQ)) * stride_ln)

        offs_q = i * BQ + tl.arange(0, BQ)
        if HAS_MASK:
            q_valid = tl.load(
                M + pid_b * stride_mb + offs_q * stride_mn,
                mask=offs_q < n,
                other=0,
            ).to(tl.int1)
        else:
            q_valid = tl.full([BQ], value=1, dtype=tl.int1)
        
        q_dtype = q_i.dtype

        # 重新计算注意力分数 S 和 Softmax 结果 P
        s_ij = tl.dot((q_i * d_scale).to(q_dtype), kt_j)
        if IS_CAUSAL:
            offs_q = i * BQ + tl.arange(0, BQ)
            offs_k = pid_tk * BK + tl.arange(0, BK)
            s_ij += tl.where(offs_q[:, None] >= offs_k[None, :], 0, float("-inf"))

        if HAS_MASK:
            offs_k = pid_tk * BK + tl.arange(0, BK)
            k_valid = tl.load(
                M + pid_b * stride_mb + offs_k * stride_mn,
                mask=offs_k < n,
                other=0,
            ).to(tl.int1)
            s_ij += tl.where(k_valid[None, :], 0.0, float("-inf"))
            
        p_ij = tl.exp(s_ij - l_i[:, None])
        if HAS_MASK:
            p_ij = tl.where(q_valid[:, None] & k_valid[None, :], p_ij, 0.0)
            q_valid_f = q_valid.to(tl.float32).to(do_i.dtype)
            do_i = do_i * q_valid_f[:, None]
            o_i = o_i * q_valid_f[:, None]
        
        # 计算 dV: dv += p_ij^T @ do_i
        dv += tl.dot(tl.trans(p_ij.to(q_dtype)), do_i)
        
        # 计算 dP: dp = do_i @ v_j^T
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        
        # 计算 Di = rowsum(DO * O)
        di = tl.sum(do_i * o_i, axis=1)
        
        # 计算 dS: ds = P * (dP - Di)
        ds_ij = p_ij * (dp_ij - di[:, None])
        ds_ij = (ds_ij * d_scale).to(q_dtype)
        
        # 计算 dK: dk += ds^T @ q_i
        dk += tl.dot(tl.trans(ds_ij), q_i)
        
        # 计算 dQ 并使用原子加法更新 (因为多个 K 块会贡献同一个 Q 块)
        dq_i = tl.dot(ds_ij, k_j)
        tl.atomic_add(DQ + pid_b * stride_qb + (i * BQ + tl.arange(0, BQ))[:, None] * stride_qn + tl.arange(0, D)[None, :] * stride_qd, dq_i.to(tl.float32))

    # 存储 dK 和 dV
    dk_block_ptr = tl.make_block_ptr(
        base=DK + pid_b * stride_kb,
        shape=(n, D),
        strides=(stride_kn, stride_kd),
        offsets=(pid_tk * BK, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        base=DV + pid_b * stride_vb,
        shape=(n, D),
        strides=(stride_vn, stride_vd),
        offsets=(pid_tk * BK, 0),
        block_shape=(BK, D),
        order=(1, 0),
    )
    tl.store(dk_block_ptr, dk.to(k_dtype), boundary_check=(0, 1))
    tl.store(dv_block_ptr, dv.to(k_dtype), boundary_check=(0, 1))

# -----------------------------------------------------------------------------
# 3. Python Wrapper & Autograd Function
# -----------------------------------------------------------------------------

def flash_attention_forward(q, k, v, causal=False, key_padding_mask=None):
    # 确保输入是连续的，以避免 view 错误
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    l = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    
    # 重新整理形状为 (B*H, N, D)
    q_flat = q.view(-1, N, D)
    k_flat = k.view(-1, N, D)
    v_flat = v.view(-1, N, D)
    o_flat = o.view(-1, N, D)
    
    d_scale = D ** -0.5
    
    # 使用 lambda 定义 grid，以便 autotune 可以根据不同的 BQ 调整 grid 大小
    grid = lambda META: (q_flat.shape[0], triton.cdiv(N, META['BQ']))
    
    if key_padding_mask is not None and key_padding_mask.numel() != 0:
        key_padding_mask = key_padding_mask.contiguous()
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask != 0
        if key_padding_mask.dim() != 2 or key_padding_mask.shape[0] != B or key_padding_mask.shape[1] != N:
            raise ValueError(f"key_padding_mask must have shape (B, N)=({B}, {N}), got {tuple(key_padding_mask.shape)}")
        mask_flat = key_padding_mask[:, None, :].expand(B, H, N).reshape(B * H, N).contiguous()
        has_mask = True
    else:
        mask_flat = torch.empty((1, 1), device=q.device, dtype=torch.int8)
        has_mask = False

    _flash_attention_fwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat, l,
        mask_flat,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        l.stride(0), l.stride(1),
        mask_flat.stride(0), mask_flat.stride(1),
        N, d_scale,
        IS_CAUSAL=causal,
        HAS_MASK=has_mask,
        D=D
    )
    return o, l

def flash_attention_backward(do, q, k, v, o, l, causal=False, key_padding_mask=None):
    # 确保输入是连续的，以避免 view 错误
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    do = do.contiguous()

    B, H, N, D = q.shape
    # 使用 float32 累加 dQ 以提高原子加法的数值稳定性
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    
    q_flat = q.view(-1, N, D)
    k_flat = k.view(-1, N, D)
    v_flat = v.view(-1, N, D)
    o_flat = o.view(-1, N, D)
    do_flat = do.view(-1, N, D)
    dq_flat = dq.view(-1, N, D)
    dk_flat = dk.view(-1, N, D)
    dv_flat = dv.view(-1, N, D)
    
    d_scale = D ** -0.5
    
    # 使用 lambda 定义 grid，以便 autotune 可以根据不同的 BK 调整 grid 大小
    grid = lambda META: (q_flat.shape[0], triton.cdiv(N, META['BK']))
    
    if key_padding_mask is not None and key_padding_mask.numel() != 0:
        key_padding_mask = key_padding_mask.contiguous()
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask != 0
        if key_padding_mask.dim() != 2 or key_padding_mask.shape[0] != B or key_padding_mask.shape[1] != N:
            raise ValueError(f"key_padding_mask must have shape (B, N)=({B}, {N}), got {tuple(key_padding_mask.shape)}")
        mask_flat = key_padding_mask[:, None, :].expand(B, H, N).reshape(B * H, N).contiguous()
        has_mask = True
    else:
        mask_flat = torch.empty((1, 1), device=q.device, dtype=torch.int8)
        has_mask = False

    _flash_attention_bwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat, l,
        mask_flat,
        do_flat, dq_flat, dk_flat, dv_flat,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        l.stride(0), l.stride(1),
        mask_flat.stride(0), mask_flat.stride(1),
        N, d_scale,
        IS_CAUSAL=causal,
        HAS_MASK=has_mask,
        D=D
    )
    return dq.to(q.dtype), dk, dv

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, key_padding_mask=None):
        # 使用 internal 算子以支持 torch.compile
        if key_padding_mask is None:
            key_padding_mask = torch.empty((0,), device=q.device, dtype=torch.int8)
        o, l = torch.ops.custom_op.flash_attention_internal(q, k, v, causal, key_padding_mask)
        ctx.save_for_backward(q, k, v, o, l, key_padding_mask)
        ctx.causal = causal
        ctx.has_mask = key_padding_mask.numel() != 0
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, key_padding_mask = ctx.saved_tensors
        mask = key_padding_mask if ctx.has_mask else torch.empty((0,), device=q.device, dtype=torch.int8)
        dq, dk, dv = torch.ops.custom_op.flash_attention_backward(do, q, k, v, o, l, ctx.causal, mask)
        return dq, dk, dv, None, None

# -----------------------------------------------------------------------------
# 4. Operator Registration & Public API
# -----------------------------------------------------------------------------

# 定义主算子
try:
    torch.library.define("custom_op::flash_attention", "(Tensor q, Tensor k, Tensor v, bool causal, Tensor key_padding_mask) -> Tensor")
    # 定义内部算子（返回 o 和 l）
    torch.library.define("custom_op::flash_attention_internal", "(Tensor q, Tensor k, Tensor v, bool causal, Tensor key_padding_mask) -> (Tensor, Tensor)")
    # 定义反向算子
    torch.library.define("custom_op::flash_attention_backward", "(Tensor do, Tensor q, Tensor k, Tensor v, Tensor o, Tensor l, bool causal, Tensor key_padding_mask) -> (Tensor, Tensor, Tensor)")
except:
    pass # 算子可能已经注册过了

@torch.library.register_fake("custom_op::flash_attention")
def flash_attention_fake(q, k, v, causal, key_padding_mask):
    return torch.empty_like(q)

@torch.library.register_fake("custom_op::flash_attention_internal")
def flash_attention_internal_fake(q, k, v, causal, key_padding_mask):
    B, H, N, D = q.shape
    return torch.empty_like(q), torch.empty((B * H, N), device=q.device, dtype=torch.float32)

@torch.library.register_fake("custom_op::flash_attention_backward")
def flash_attention_backward_fake(do, q, k, v, o, l, causal, key_padding_mask):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

@torch.library.register_kernel("custom_op::flash_attention_internal", "cuda")
def flash_attention_internal_cuda(q, k, v, causal, key_padding_mask):
    return flash_attention_forward(q, k, v, causal, key_padding_mask=key_padding_mask)

@torch.library.register_kernel("custom_op::flash_attention_backward", "cuda")
def flash_attention_backward_cuda(do, q, k, v, o, l, causal, key_padding_mask):
    return flash_attention_backward(do, q, k, v, o, l, causal, key_padding_mask=key_padding_mask)

@torch.library.register_kernel("custom_op::flash_attention", "Autograd")
def flash_attention_autograd(q, k, v, causal, key_padding_mask):
    return FlashAttention.apply(q, k, v, causal, key_padding_mask)

def flash_attention_fn(q, k, v, causal=False, key_padding_mask=None):
    """
    Flash Attention 接口，支持 torch.compile。
    q, k, v: (B, H, N, D)
    """
    if key_padding_mask is None:
        key_padding_mask = torch.empty((0,), device=q.device, dtype=torch.int8)
    return torch.ops.custom_op.flash_attention(q, k, v, causal, key_padding_mask)
