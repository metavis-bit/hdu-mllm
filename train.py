import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from loader import load_qwen3_custom
from inference import _count_markers, _find_marker_end_token_indices, _last_user_region


def passthrough_collate(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return samples


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON array file must contain a list of objects.")
            return data
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


class JsonMultimodalDataset(Dataset):
    def __init__(self, json_path: str):
        self.records = _read_json_records(json_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        if not isinstance(r, dict):
            raise ValueError(f"Record {idx} must be an object/dict.")

        user = r.get("user")
        assistant = r.get("assistant")
        if user is None or assistant is None:
            raise ValueError(f"Record {idx} missing user/assistant.")

        images = r.get("image_paths") or r.get("images") or r.get("image")
        
        image_paths: List[str] = []
        if images is not None:
            if isinstance(images, str):
                image_paths = [images]
            elif isinstance(images, list):
                image_paths = [str(p) for p in images]
            else:
                raise ValueError(f"Record {idx} image_paths/images/image has unsupported type: {type(images)}")
        
        image_tensors = torch.empty((0, 3, 224, 224))
        if image_paths:
            image_tensors = _load_images(image_paths)

        return {
            "user": str(user), 
            "assistant": str(assistant), 
            "image_paths": image_paths,
            "image_tensors": image_tensors,
            "image_count": len(image_paths)
        }


def _load_images(image_paths: Sequence[str]) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    images: List[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((1024, 1024))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        images.append(t)
    return torch.stack(images, dim=0)


@dataclass
class Batch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class MultimodalCollator:
    def __init__(
        self,
        tokenizer,
        device: str,
        model_dtype: torch.dtype,
        eos_token_id: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.model_dtype = model_dtype
        self.eos_token_id = eos_token_id

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return samples

    def build_batch(self, model, samples: List[Dict[str, Any]]) -> Batch:
        full_prompts: List[str] = []
        assistant_regions: List[Tuple[int, int]] = []
        per_sample_image_paths: List[List[str]] = []
        for s in samples:
            user_text = str(s["user"])
            assistant_text = str(s["assistant"])
            image_paths = list(s.get("image_paths") or [])
            if image_paths:
                expected = len(image_paths)
                count = _count_markers(user_text, "<|vision_start|>")
                if count == 0:
                    user_text = "<|vision_start|><|vision_end|>" * expected + (" " + user_text if user_text else "")
                else:
                    if count != expected:
                        raise ValueError(
                            f"image_paths={expected} but <|vision_start|> count in sample user={count}."
                        )

            messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]
            try:
                full_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            except TypeError:
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # HACK: Remove empty <think> blocks if the template inserts them but the model doesn't support them
            # full_prompt = full_prompt.replace("<think>\n\n</think>\n\n", "")

            a_start_tag = "<|im_start|>assistant\n"
            a_end_tag = "<|im_end|>"
            a_start = full_prompt.find(a_start_tag)
            if a_start < 0:
                raise ValueError("Failed to locate assistant start tag in chat template.")
            a_start += len(a_start_tag)
            a_end = full_prompt.find(a_end_tag, a_start)
            if a_end < 0:
                raise ValueError("Failed to locate assistant end tag in chat template.")
            full_prompts.append(full_prompt)
            assistant_regions.append((a_start, a_end))
            per_sample_image_paths.append(image_paths)

        encodings = []
        for p in full_prompts:
            try:
                enc = self.tokenizer(p, add_special_tokens=False, return_offsets_mapping=True)
                encodings.append(enc)
            except Exception:
                enc = self.tokenizer(p, add_special_tokens=False)
                enc["offset_mapping"] = None
                encodings.append(enc)

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        marker_ends_list: List[List[int]] = []
        for p, a_region, enc, img_paths in zip(full_prompts, assistant_regions, encodings, per_sample_image_paths):
            ids = enc["input_ids"]
            input_ids = torch.tensor(ids, dtype=torch.long)
            input_ids_list.append(input_ids)

            offset_mapping = enc.get("offset_mapping")
            if offset_mapping is not None:
                offset_mapping = list(offset_mapping)
            a_start, a_end = a_region
            a_end_with_tag = a_end + len(a_end_tag)
            
            labels = torch.full_like(input_ids, -100)
            if offset_mapping is not None:
                for ti, (s, e) in enumerate(offset_mapping):
                    if s == 0 and e == 0:
                        continue
                    # 包含回复内容和结束标签
                    if s < a_end_with_tag and e > a_start:
                        labels[ti] = input_ids[ti]
            else:
                prefix_ids = self.tokenizer(p[:a_start], add_special_tokens=False).input_ids
                end_ids = self.tokenizer(p[:a_end_with_tag], add_special_tokens=False).input_ids
                lo = len(prefix_ids)
                hi = len(end_ids)
                lo = max(0, min(lo, labels.shape[0]))
                hi = max(lo, min(hi, labels.shape[0]))
                if hi > lo:
                    labels[lo:hi] = input_ids[lo:hi]
            labels_list.append(labels)

            user_region = _last_user_region(p)
            marker_ends = _find_marker_end_token_indices(
                p,
                input_ids.tolist(),
                self.tokenizer,
                marker="<|vision_start|>",
                offset_mapping=offset_mapping,
                search_region=user_region,
            )
            if img_paths and len(marker_ends) != len(img_paths):
                raise ValueError(f"<|vision_start|> count ({len(marker_ends)}) must match image_paths ({len(img_paths)}).")
            marker_ends_list.append(marker_ends)

        image_tokens_per_sample: List[List[torch.Tensor]] = [[] for _ in samples]
        image_counts: List[int] = []
        flat_images_cpu: List[torch.Tensor] = []
        for s in samples:
            image_tensors = s.get("image_tensors")
            if image_tensors is not None and image_tensors.numel() != 0:
                if image_tensors.dim() != 4 or image_tensors.shape[1] != 3:
                    raise ValueError(f"image_tensors must have shape (N, 3, H, W), got {tuple(image_tensors.shape)}")
                image_counts.append(int(image_tensors.shape[0]))
                flat_images_cpu.append(image_tensors)
            else:
                image_counts.append(0)

        total_images = sum(image_counts)
        if total_images > 0:
            flat_images = torch.cat(flat_images_cpu, dim=0).to(
                device=self.device, dtype=self.model_dtype, non_blocking=True
            )
            image_tokens_all = model.encode_images(flat_images)
            if isinstance(image_tokens_all, tuple):
                image_tokens_all, _ = image_tokens_all
            if image_tokens_all.shape[0] != total_images:
                raise RuntimeError(
                    f"encode_images returned {image_tokens_all.shape[0]} images, expected {total_images}."
                )
            ptr = 0
            for bi, cnt in enumerate(image_counts):
                if cnt > 0:
                    toks = image_tokens_all[ptr : ptr + cnt]
                    for i in range(toks.shape[0]):
                        image_tokens_per_sample[bi].append(toks[i])
                    ptr += cnt

        seq_embeds: List[torch.Tensor] = []
        seq_labels: List[torch.Tensor] = []
        seq_masks: List[torch.Tensor] = []

        for bi, (input_ids, labels, marker_ends, img_tokens_list) in enumerate(
            zip(input_ids_list, labels_list, marker_ends_list, image_tokens_per_sample)
        ):
            input_ids = input_ids.to(device=self.device, non_blocking=True)
            text_embeds = model.model.embed_tokens(input_ids.unsqueeze(0))
            
            labels = labels.to(device=self.device, non_blocking=True)
            mask = torch.ones_like(labels, dtype=torch.int8)

            if img_tokens_list:
                if len(img_tokens_list) != len(marker_ends):
                    raise RuntimeError("Image/token alignment mismatch within sample.")
                inserts = []
                for end_idx, img_tok in zip(marker_ends, img_tokens_list):
                    inserts.append((end_idx + 1, img_tok))
                inserts.sort(key=lambda x: x[0], reverse=True)

                inputs_embeds = text_embeds
                labels_seq = labels
                mask_seq = mask
                for pos, tok in inserts:
                    tok = tok.to(device=self.device, dtype=self.model_dtype, non_blocking=True)
                    tok_len = tok.shape[0]
                    inputs_embeds = torch.cat(
                        [inputs_embeds[:, :pos, :], tok.unsqueeze(0), inputs_embeds[:, pos:, :]],
                        dim=1,
                    )
                    img_labels = torch.full((tok_len,), -100, device=labels_seq.device, dtype=labels_seq.dtype)
                    img_mask = torch.ones((tok_len,), device=mask_seq.device, dtype=mask_seq.dtype)
                    labels_seq = torch.cat([labels_seq[:pos], img_labels, labels_seq[pos:]], dim=0)
                    mask_seq = torch.cat([mask_seq[:pos], img_mask, mask_seq[pos:]], dim=0)
                    if not torch.equal(inputs_embeds[:, pos : pos + tok_len, :], tok.unsqueeze(0)):
                        raise RuntimeError("Image token insertion check failed.")
                text_embeds = inputs_embeds
                labels = labels_seq
                mask = mask_seq

            seq_embeds.append(text_embeds.squeeze(0))
            seq_labels.append(labels)
            seq_masks.append(mask)

        max_len = max(x.shape[0] for x in seq_labels)
        hidden = seq_embeds[0].shape[-1]

        batch_embeds = torch.zeros((len(samples), max_len, hidden), device=self.device, dtype=self.model_dtype)
        batch_labels = torch.full((len(samples), max_len), -100, device=self.device, dtype=torch.long)
        batch_mask = torch.zeros((len(samples), max_len), device=self.device, dtype=torch.int8)

        for i, (emb, lab, msk) in enumerate(zip(seq_embeds, seq_labels, seq_masks)):
            L = lab.shape[0]
            batch_embeds[i, :L] = emb.to(device=self.device, dtype=self.model_dtype, non_blocking=True)
            batch_labels[i, :L] = lab.to(device=self.device, dtype=torch.long, non_blocking=True)
            batch_mask[i, :L] = msk.to(device=self.device, dtype=torch.int8, non_blocking=True)

        return Batch(inputs_embeds=batch_embeds, attention_mask=batch_mask, labels=batch_labels)


def train(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = load_qwen3_custom(args.model_dir, device=device, dtype=torch.bfloat16)

    for name, param in model.named_parameters():
        if ("vision_backbone" in name) or ("vision_projector" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 在所有梯度设置完成后，再统计参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable_params:,}")

    # 统一打印所有可训练参数，让你看得清楚
    print("--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training: {name}")
    print("----------------------------")

    eos_token_id = tokenizer.eos_token_id
    dtype = next(model.parameters()).dtype
    dataset = JsonMultimodalDataset(args.data)
    collator = MultimodalCollator(tokenizer, device=device, model_dtype=dtype, eos_token_id=eos_token_id)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=passthrough_collate,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Calculate effective total steps for LR scheduler and progress tracking
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_for_step(step: int) -> float:
        if warmup_steps <= 0:
            return args.lr
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        return args.lr

    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for i, samples in enumerate(loader):
            batch = collator.build_batch(model, samples)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch.labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum

            if loss.requires_grad:
                loss.backward()
            else:
                if i == 0: # Only print once to avoid spam
                    print(f"Warning: loss does not require grad. Check if parameters are trainable.")

            # Perform update every grad_accum steps or at the end of epoch
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update LR for this step
                lr = lr_for_step(step)
                for pg in optim.param_groups:
                    pg["lr"] = lr
                
                optim.step()
                optim.zero_grad(set_to_none=True)

                if step % args.log_every == 0:
                    # Show original loss value in logs
                    print(f"step={step} loss={loss.item() * args.grad_accum:.4f} lr={lr:.2e}")

                if args.save_dir and args.save_every > 0 and step > 0 and step % args.save_every == 0:
                    save_checkpoint(model, optim, step, args.save_dir)

                step += 1
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        
        if args.max_steps > 0 and step >= args.max_steps:
            break
    
    # End of training save
    if args.save_dir:
        save_checkpoint(model, optim, step, args.save_dir, is_final=True)

def save_checkpoint(model, optim, step, save_dir, is_final=False):
    os.makedirs(save_dir, exist_ok=True)
    if is_final:
        ckpt_name = "checkpoint-final.pt"
    else:
        ckpt_name = f"checkpoint-{step}.pt"
    
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    model_state = model.state_dict()
    for k, v in model_state.items():
        model_state[k] = v.to(torch.bfloat16)

    print(f"Saving checkpoint to {ckpt_path} (bfloat16)...")
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optim": optim.state_dict(),
        },
        ckpt_path,
    )
    print("Done.")



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=0)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
