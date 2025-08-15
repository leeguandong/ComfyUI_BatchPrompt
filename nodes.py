import json
import torch
import torch.nn.functional as F

"""
最简批量帧文本编码（仅正向），增强兼容性：
- 解决：不同 prompt 生成的 cond 长度不一致导致 torch.cat 报错 -> 统一 padding
- 解决：某些 clip.encode_from_tokens 在 return_pooled=True 时返回 (cond, None) 或根本不支持 pooled 的情况
- 若 pooled 全部为 None，则不放 pooled_output 或放占位空字符串
"""

def parse_frame_prompts(raw_text: str):
    """
    解析输入为 {frame_index: prompt} 的升序字典。
    允许缺少最外层 {}；允许末尾多余逗号。
    """
    txt = raw_text.strip()
    if not txt:
        return {}
    if not txt.startswith("{"):
        txt = "{" + txt
    if not txt.endswith("}"):
        txt = txt + "}"
    txt = txt.replace(",}", "}")
    data = json.loads(txt)
    out = {}
    for k, v in data.items():
        try:
            idx = int(k)
        except:
            continue
            # 非数字 key 忽略
        out[idx] = str(v)
    return dict(sorted(out.items(), key=lambda x: x[0]))


def encode_prompts_sequential(clip, ordered_items):
    """
    逐条编码；对 cond 做长度统一；pooled 可能为 None。
    ordered_items: List[(frame_index, prompt)]
    返回: cond_final, pooled_final(或 None), has_pooled(bool)
    """
    cond_list = []
    pooled_list = []
    pooled_available = True  # 先假设有 pooled
    max_tokens = 0

    with torch.no_grad():
        for frame_idx, prompt in ordered_items:
            tokens = clip.tokenize(prompt)
            # 兼容：有的实现如果 return_pooled=True 仍返回单值
            enc = clip.encode_from_tokens(tokens, return_pooled=True)
            if isinstance(enc, (list, tuple)) and len(enc) == 2:
                cond, pooled = enc
            else:
                cond, pooled = enc, None

            if cond.dim() != 3:
                # 期望 shape: [B(=1), T, C]; 若是 [T, C] 补一个 batch 维
                if cond.dim() == 2:
                    cond = cond.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected cond tensor shape: {cond.shape}")

            if cond.shape[1] > max_tokens:
                max_tokens = cond.shape[1]

            cond_list.append(cond)

            if pooled is None:
                pooled_available = False
                pooled_list.append(None)
            else:
                # 期望 [B(=1), D] 或 [D]; 若 [D] -> [1,D]
                if pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)
                pooled_list.append(pooled)

    if not cond_list:
        dummy = torch.zeros(1, 1)
        return dummy, None, False

    # 对 cond 做 padding
    padded = []
    for c in cond_list:
        if c.shape[1] < max_tokens:
            pad_len = max_tokens - c.shape[1]
            # shape [1, T, C]，pad 第二维 (tokens)：(last_dim_left, last_dim_right, second_dim_left, second_dim_right)
            c = F.pad(c, (0, 0, 0, pad_len))
        padded.append(c)
    cond_final = torch.cat(padded, dim=0)  # [N, max_tokens, C]

    pooled_final = None
    if pooled_available:
        # 校验 pooled 维度并 cat
        processed = []
        for p in pooled_list:
            if p is None:
                pooled_available = False
                break
            if p.dim() == 1:
                p = p.unsqueeze(0)
            processed.append(p)
        if pooled_available and processed:
            try:
                pooled_final = torch.cat(processed, dim=0)  # [N, D]
            except Exception:
                # 如果维度不一致，放弃 pooled
                pooled_final = None
                pooled_available = False

    return cond_final, pooled_final, pooled_available


class BatchPrompt:
    """
    最简批量帧文本编码节点（仅正向；start_frame/end_frame 必填）
    区间采用包含式 [start_frame, end_frame]；若 end_frame < start_frame 则自动交换。
    不做去重、不做批量 token、保持逻辑最简单。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": '"0":"A cat","1":"A dog","2":"A bird"'
                }),
                "clip": ("CLIP",),
                "start_frame": ("INT", {"default": 0, "min": 0, "step": 1}),
                "end_frame": ("INT", {"default": 2, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("COND",)
    FUNCTION = "run"
    CATEGORY = "SimpleBatch"

    def run(self, text, clip, start_frame, end_frame):
        frame_prompts = parse_frame_prompts(text)

        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        # 过滤帧（包含式）
        selected = [(i, p) for i, p in frame_prompts.items()
                    if start_frame <= i <= end_frame]

        if not selected:
            dummy = torch.zeros(1, 1)
            return ([[dummy, {"pooled_output": ""}]],)

        cond, pooled, has_pooled = encode_prompts_sequential(clip, selected)

        meta = {}
        if has_pooled and pooled is not None:
            meta["pooled_output"] = pooled
        else:
            # 与早期代码兼容，可放空字符串或直接不放
            meta["pooled_output"] = ""

        return ([[cond, meta]],)


# ComfyUI 节点注册映射
NODE_CLASS_MAPPINGS = {
    "BatchPrompt": BatchPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchPrompt": "Batch Prompt"
}
