import torch
import torch.nn.functional as F
from loader import load_qwen3_custom
from transformers import AutoTokenizer
import os
import sys
import io
from typing import Optional, Sequence, List, Tuple

def sample(logits, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, input_ids=None):
    """
    Apply temperature, top-k, top-p sampling, and repetition penalty to logits.
    """
    # 0. Apply Repetition Penalty
    if repetition_penalty != 1.0 and input_ids is not None:
        # Get unique tokens in the input_ids
        # input_ids: (1, seqlen) or (seqlen,)
        if input_ids.dim() == 2:
            prev_tokens = input_ids[0].unique()
        else:
            prev_tokens = input_ids.unique()
            
        for token_id in prev_tokens:
            # If the logit is positive, divide by penalty. If negative, multiply by penalty.
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    # 1. Apply Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # 2. Apply Top-K
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # 3. Apply Top-P (Nucleus Sampling)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices to original logits shape
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    # 4. Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def _load_images_from_paths(image_paths: Sequence[str], device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
    from PIL import Image
    import numpy as np
    import os

    if not image_paths:
        return None

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    images: List[torch.Tensor] = []
    for p in image_paths:
        if not p or not isinstance(p, str):
            print(f"Warning: Invalid image path: {p}")
            continue
        if not os.path.exists(p):
            print(f"Warning: Image path does not exist: {p}")
            continue
        try:
            img = Image.open(p).convert("RGB").resize((224, 224))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            images.append(t)
        except Exception as e:
            print(f"Error loading image {p}: {e}")
            continue

    if not images:
        return None

    return torch.stack(images, dim=0).to(device=device, dtype=dtype)

def _find_marker_end_token_indices(
    full_prompt: str,
    input_ids: List[int],
    tokenizer,
    marker: str = "<|vision_start|>",
    offset_mapping: Optional[List[Tuple[int, int]]] = None,
    search_region: Optional[Tuple[int, int]] = None,
) -> List[int]:
    if offset_mapping is not None:
        ends: List[int] = []
        start = 0 if search_region is None else search_region[0]
        end = None if search_region is None else search_region[1]
        while True:
            idx = full_prompt.find(marker, start) if end is None else full_prompt.find(marker, start, end)
            if idx < 0:
                break
            end_char = idx + len(marker)
            found = None
            for ti, (s, e) in enumerate(offset_mapping):
                if s == 0 and e == 0:
                    continue
                if s < end_char <= e:
                    found = ti
                    break
            if found is None:
                raise ValueError(f"Failed to align marker '{marker}' to token offsets.")
            ends.append(found)
            start = end_char
        return ends

    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    if not marker_ids:
        return []
    ends = []
    i = 0
    while i <= len(input_ids) - len(marker_ids):
        if input_ids[i : i + len(marker_ids)] == marker_ids:
            ends.append(i + len(marker_ids) - 1)
            i += len(marker_ids)
        else:
            i += 1
    return ends

def _last_user_region(full_prompt: str) -> Optional[Tuple[int, int]]:
    user_tag = "<|im_start|>user\n"
    end_tag = "<|im_end|>"
    start = full_prompt.rfind(user_tag)
    if start < 0:
        return None
    start += len(user_tag)
    end = full_prompt.find(end_tag, start)
    if end < 0:
        return None
    return (start, end)

def _count_markers(text: str, marker: str) -> int:
    return text.count(marker)

def inference(
    model,
    tokenizer,
    messages,
    device="cuda",
    max_new_tokens=512,
    temperature=0.7,
    top_k=10,
    top_p=0.3,
    repetition_penalty=1.1,
    enable_thinking=False,
    stream=False,
    image_paths: Optional[Sequence[str]] = None,
):
    # Limit to last 20 rounds (1 round = user + assistant, so 40 messages)
    # Keep the system message if it exists
    if len(messages) > 0:
        system_msg = [messages[0]] if messages[0].get("role") == "system" else []
        other_msgs = messages[1:] if system_msg else messages
        
        # 20 rounds * 2 = 40 messages
        max_history = 40
        if len(other_msgs) > max_history:
            other_msgs = other_msgs[-max_history:]
            # Ensure we start with a user message for better template compatibility
            while other_msgs and other_msgs[0].get("role") != "user":
                other_msgs = other_msgs[1:]
            
        messages_for_prompt = system_msg + other_msgs
    else:
        messages_for_prompt = messages

    if image_paths:
        last_user_idx = None
        for i in range(len(messages_for_prompt) - 1, -1, -1):
            if messages_for_prompt[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            raise ValueError("image_paths provided but no user message found.")
        user_content = str(messages_for_prompt[last_user_idx].get("content", ""))
        expected = len(image_paths)
        count = _count_markers(user_content, "<|vision_start|>")
        if count == 0:
            patched = "<|vision_start|><|vision_end|>" * expected + (" " + user_content if user_content else "")
            messages_for_prompt = [dict(m) for m in messages_for_prompt]
            messages_for_prompt[last_user_idx] = dict(messages_for_prompt[last_user_idx])
            messages_for_prompt[last_user_idx]["content"] = patched
        elif count != expected:
            raise ValueError(
                f"image_paths={expected} but <|vision_start|> count in last user message={count}."
            )

    def _stream():
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages_for_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            full_prompt = tokenizer.apply_chat_template(messages_for_prompt, tokenize=False, add_generation_prompt=True)

        try:
            encoding = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
            offset_mapping = encoding.offset_mapping[0].tolist()
        except Exception:
            encoding = tokenizer(full_prompt, return_tensors="pt")
            offset_mapping = None

        input_ids = encoding.input_ids.to(device)
        if input_ids.shape[1] == 0:
            return

        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None:
            stop_token_ids.append(im_end_id)

        with torch.no_grad():
            if image_paths:
                region = _last_user_region(full_prompt)
                marker_ends = _find_marker_end_token_indices(
                    full_prompt,
                    input_ids[0].tolist(),
                    tokenizer,
                    marker="<|vision_start|>",
                    offset_mapping=offset_mapping,
                    search_region=region,
                )
                if len(marker_ends) != len(image_paths):
                    raise ValueError(
                        f"<|vision_start|> count ({len(marker_ends)}) must match image_paths ({len(image_paths)})."
                    )

                param = next(model.parameters(), None)
                model_dtype = param.dtype if param is not None else torch.float32
                images = _load_images_from_paths(image_paths, device=device, dtype=model_dtype)
                if images is not None:
                    image_tokens = model.encode_images(images)
                else:
                    image_tokens = torch.zeros((0, 0, 0), device=device, dtype=model_dtype)
                text_embeds = model.model.embed_tokens(input_ids)

                inserts = []
                for img_i, end_idx in enumerate(marker_ends):
                    inserts.append((end_idx + 1, image_tokens[img_i : img_i + 1]))
                inserts.sort(key=lambda x: x[0], reverse=True)

                inputs_embeds = text_embeds
                for pos, tok in inserts:
                    inputs_embeds = torch.cat(
                        [inputs_embeds[:, :pos, :], tok, inputs_embeds[:, pos:, :]],
                        dim=1,
                    )
                    if not torch.equal(inputs_embeds[:, pos : pos + tok.shape[1], :], tok):
                        raise RuntimeError("Image token insertion check failed.")

                logits, past_key_values = model.generate_step(
                    inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True
                )
                curr_input_ids_all = input_ids # Start tracking all tokens for penalty
            else:
                logits, past_key_values = model.generate_step(input_ids, past_key_values=None, use_cache=True)
                curr_input_ids_all = input_ids

            curr_input_ids = sample(
                logits[:, -1, :], 
                temperature, 
                top_k, 
                top_p, 
                repetition_penalty=repetition_penalty, 
                input_ids=curr_input_ids_all
            )
            step_count = 0
            all_new_tokens = []
            printable_text = ""
            while True:
                token_id = int(curr_input_ids.item())
                if token_id in stop_token_ids:
                    break

                all_new_tokens.append(token_id)
                full_text = tokenizer.decode(all_new_tokens, skip_special_tokens=True)
                
                # Check if the text is complete (not ending with replacement character)
                # and if we have new content to yield
                if not full_text.endswith('\ufffd') and len(full_text) > len(printable_text):
                    new_text = full_text[len(printable_text):]
                    yield new_text
                    printable_text = full_text

                step_count += 1
                if step_count >= max_new_tokens:
                    break

                # Update tracked tokens
                curr_input_ids_all = torch.cat([curr_input_ids_all, curr_input_ids], dim=1)
                
                logits, past_key_values = model.generate_step(
                    curr_input_ids, past_key_values=past_key_values, use_cache=True
                )
                curr_input_ids = sample(
                    logits[:, -1, :], 
                    temperature, 
                    top_k, 
                    top_p, 
                    repetition_penalty=repetition_penalty, 
                    input_ids=curr_input_ids_all
                )
            
            # Yield any remaining text after the loop
            full_text = tokenizer.decode(all_new_tokens, skip_special_tokens=True)
            if len(full_text) > len(printable_text):
                yield full_text[len(printable_text):]

    if stream:
        return _stream()
    return "".join(list(_stream()))


def realtime_chat(
    model,
    tokenizer,
    device="cuda",
    system_prompt="You are a helpful assistant.",
    max_new_tokens=20000,
    temperature=0.7,
    top_k=10,
    top_p=0.3,
    repetition_penalty=1.1,
    enable_thinking=True,
):
    print("\n" + "=" * 50)
    print("Qwen3 Custom Chat Mode (Stateless Across Turns)")
    print(f"Sampling: Temp={temperature}, TopK={top_k}, TopP={top_p}, Penalty={repetition_penalty}, Thinking={enable_thinking}")
    print("=" * 50)

    messages = [{"role": "system", "content": system_prompt}]
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})
            print("Assistant: ", end="", flush=True)

            chunks = []
            for chunk in inference(
                model,
                tokenizer,
                messages,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
                stream=True,
            ):
                chunks.append(chunk)
                print(chunk, end="", flush=True)

            assistant_text = "".join(chunks)
            messages.append({"role": "assistant", "content": assistant_text})
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    model_dir = r"D:\d2l-zh\Mamba\Qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 2. Load Custom Model
    print("Loading custom model...")
    # Using bfloat16 for better performance/accuracy on modern GPUs
    model = load_qwen3_custom(model_dir, device=device, dtype=torch.bfloat16)
    
    # 3. Start Chat
    realtime_chat(model, tokenizer, device=device)
