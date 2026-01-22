import torch
import json
import os
from safetensors.torch import load_file
from model import Qwen3ForCausalLM

def load_qwen3_custom(model_path, device="cpu", dtype=torch.bfloat16):
    # 1. Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    
    # 2. Initialize custom model
    print(f"Initializing custom Qwen3 model with config: {config['model_type']}...")
    model = Qwen3ForCausalLM(config).to(dtype)
    
    # 3. Load weight map from index
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # 4. Load weights from safetensors files
    # Group weights by file to minimize loading calls
    files_to_load = {}
    for weight_name, filename in weight_map.items():
        if filename not in files_to_load:
            files_to_load[filename] = []
        files_to_load[filename].append(weight_name)
    
    state_dict = {}
    for filename, weights in files_to_load.items():
        print(f"Loading weights from {filename}...")
        shard_path = os.path.join(model_path, filename)
        shard_state_dict = load_file(shard_path)
        
        for weight_name in weights:
            if weight_name in shard_state_dict:
                state_dict[weight_name] = shard_state_dict[weight_name].to(dtype)
    
    # 5. Load state dict into model
    print("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
        
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    # Test loading
    model_dir = r"D:\d2l-zh\Mamba\Qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = load_qwen3_custom(model_dir, device=device)
        print("Model loaded successfully!")
        
        # Simple inference test
        test_input = torch.tensor([[1, 2, 3]]).to(device)
        with torch.no_grad():
            logits = model(test_input)
            print(f"Output logits shape: {logits.shape}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
