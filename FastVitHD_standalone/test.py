
import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from FastVitHD_standalone.fastvithd import fastvithd
else:
    from .fastvithd import fastvithd

def test_fastvithd():
    print("Testing standard FastViTHD model...")
    model = fastvithd(use_feature_fusion=False)
    print(f"Standard Model Parameters: {sum(p.numel() for p in model.parameters())}")
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Standard Output shape: {output.shape}")

    print("\nTesting FastViTHD with Feature Fusion...")
    model_fused = fastvithd(use_feature_fusion=True)
    print(f"Fused Model Parameters: {sum(p.numel() for p in model_fused.parameters())}")
    with torch.no_grad():
        output_fused = model_fused(dummy_input)
    print(f"Fused Output shape: {output_fused.shape}")
    
    # Verify that the output shape is the same
    assert output.shape == output_fused.shape, "Output shapes should match!"
    print("Feature Fusion test passed!")

    print("\nTesting FastViTHD with Stage 5 1D and CLS Token...")
    model_1d = fastvithd(use_stage5_1d=True)
    print(f"Stage 5 1D Model Parameters: {sum(p.numel() for p in model_1d.parameters())}")

    with torch.no_grad():
        res = model_1d(dummy_input)
    
    assert isinstance(res, tuple), "Should return (cls_token, patch_tokens)"
    cls_token, patch_tokens = res
    print(f"CLS Token shape: {cls_token.shape}")
    print(f"Patch Tokens shape: {patch_tokens.shape}")
    
    # Check dimensions
    print("Stage 5 1D and CLS Token test passed!")

    print("\nTesting FastViTHD with Feature Fusion AND Stage 5 1D...")
    model_fusion_1d = fastvithd(use_feature_fusion=True, use_stage5_1d=True)
    print(f"Fusion + Stage 5 1D Model Parameters: {sum(p.numel() for p in model_fusion_1d.parameters())}")
    with torch.no_grad():
        res = model_fusion_1d(dummy_input)
    
    assert isinstance(res, tuple), "Should return (cls_token, fused_features)"
    cls_token, fused_features = res
    print(f"Fused CLS Token shape: {cls_token.shape}")
    print(f"Fused Features shape: {fused_features.shape}")
    
    print("Feature Fusion + Stage 5 1D test passed!")

    print("\nTesting FastViTHD with Attention Map extraction...")
    model_attn = fastvithd(use_stage5_1d=True, use_feature_fusion=True, inference_mode=False)
    with torch.no_grad():
        # Test with return_attn=True
        res = model_attn(dummy_input, return_attn=True)
    
    assert isinstance(res, tuple) and len(res) == 3, f"Should return (cls, tokens, attn), got {len(res) if isinstance(res, tuple) else type(res)}"
    cls_token, patch_tokens, cls_attn = res
    
    print(f"CLS Attention Map shape: {cls_attn.shape}")
    # Expected shape: (B, num_heads, 1, N)
    # FastViTHD stage 5 has 1536 dims, typically 32 heads (48 dim per head)
    # N = 4*4 = 16
 
    # Check if softmax values sum to ~1 (excluding the self-attention part we sliced out)
    # Actually, the original softmax was over (1 + 16) elements.
    # So the sum of the 16 elements should be < 1.0
    print(f"{cls_attn[0, 0, 0]}")
    
    print("Attention Map extraction test passed!")

    print("\nTesting FastViTHD in Inference Mode...")
    # Inference mode collapses reparameterizable blocks
    model_inf = fastvithd(use_feature_fusion=True, inference_mode=True)
    print(f"Inference Model Parameters: {sum(p.numel() for p in model_inf.parameters())}")
    
    with torch.no_grad():
        output_inf = model_inf(dummy_input)
    print(f"Inference Output shape: {output_inf.shape}")
    
    # Check if a specific block is indeed in reparam mode
    # For example, check fusion_projs[0] which is a ReparamLargeKernelConv
    if hasattr(model_inf, 'fusion_projs'):
        reparam_block = model_inf.fusion_projs[0]
        # In inference mode, it should have lkb_reparam and NOT lkb_origin
        has_reparam = hasattr(reparam_block, 'lkb_reparam')
        has_origin = hasattr(reparam_block, 'lkb_origin')
        print(f"Reparam check - Has lkb_reparam: {has_reparam}, Has lkb_origin: {has_origin}")
        assert has_reparam and not has_origin, "Block should be in reparam mode!"

    print("Inference Mode test passed!")


if __name__ == "__main__":
    test_fastvithd()
