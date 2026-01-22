# hdu-mllm
hdu-mllm

## 训练阶段视觉 token 插入
训练阶段的视觉 token 插入逻辑是手写实现的，位置在 `train.py` 的 `MultimodalCollator.build_batch`：
- 先调用 `model.encode_images` 得到图像 token。
- 再通过 `torch.cat` 按 `<|vision_start|>` 标记位置把图像 token 拼接到文本 embedding 中。
- 同步扩展 labels 与 attention mask。
- 当前没有使用额外的封装/第三方插入方法。
