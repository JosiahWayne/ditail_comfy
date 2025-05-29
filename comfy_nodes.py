import os
import torch
import numpy as np
from types import SimpleNamespace
from PIL import Image
from .ditail_demo import DitailDemo
from .ditail_utils import seed_everything


class DitailStyleTransfer:
    CATEGORY = "Ditail"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # 模型
            "source_model": ("STRING", {"default": "stablediffusionapi/realistic-vision-v51"}),
            "target_model": ("STRING", {"default": "stablediffusionapi/realistic-vision-v51"}),

            # 原图
            "image":        ("IMAGE",),
            # 随机种子
            "seed":         ("INT",    {"default": 42}),
            # 正/负 prompt
            "pos_prompt":   ("STRING", {"default": "a photo"}),
            "neg_prompt":   ("STRING", {"default": "worst quality, blurry, NSFW"}),
            # 反演/采样步数
            "inv_steps":    ("INT",    {"default": 50, "min": 1, "max": 200}),
            "spl_steps":    ("INT",    {"default": 50, "min": 1, "max": 200}),
            # 权重系数
            "alpha":        ("FLOAT",  {"default": 2.0, "min": 0.0, "max": 10.0}),
            "beta":         ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 10.0}),
            "omega":        ("FLOAT",  {"default": 15.0,"min": 0.0, "max": 50.0}),
            # PnP 注入参数
            "attn_ratio":   ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 1.0}),
            "conv_ratio":   ("FLOAT",  {"default": 0.8, "min": 0.0, "max": 1.0}),
            "mask_type":    ("STRING", {"default": "full"}),
            # LoRA 支持
            "lora":         ("STRING", {"default": "none"}),
            "lora_dir":     ("STRING", {"default": "./lora"}),
            "lora_scale":   ("FLOAT",  {"default": 0.7, "min": 0.0, "max": 5.0}),
        }}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("result_image",)
    FUNCTION      = "run"

    @staticmethod
    def tensor_to_pil(t):
        t = t[0].clamp(0, 1) if isinstance(t, torch.Tensor) else t[0]
        arr = (t * 255).byte().cpu().numpy() if isinstance(t, torch.Tensor) \
              else (t * 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def pil_to_tensor(img: Image.Image, dtype=torch.float32, device="cpu"):
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).to(device, dtype)
        return tensor.unsqueeze(0)

    def run(self, source_model, target_model, image,
            seed, pos_prompt, neg_prompt,
            inv_steps, spl_steps,
            alpha, beta, omega,
            attn_ratio, conv_ratio, mask_type,
            lora, lora_dir, lora_scale,
            no_injection=False):

        # 1) 设定种子
        seed_everything(seed)
        # print(dir(source_model.patch_model()), dir(source_model.patch_model().name))
        # 设定采样器)
        # 2) 准备临时目录
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)

        # 3) Tensor → PIL → 存盘
        pil_img    = DitailStyleTransfer.tensor_to_pil(image)
        in_path    = os.path.join(temp_dir, "input.png")
        pil_img.save(in_path)

        # 4) 构造 args
        device_str = "cuda" if torch.cuda.is_available() else "mps"
        args = SimpleNamespace(
            seed        = seed,
            device      = device_str,
            output_dir  = temp_dir,
            inv_model   = source_model,
            spl_model   = target_model,
            img_path    = in_path,
            pos_prompt  = pos_prompt,
            neg_prompt  = neg_prompt,
            alpha       = alpha,
            beta        = beta,
            omega       = omega,
            inv_steps   = inv_steps,
            spl_steps   = spl_steps,
            mask        = mask_type,
            lora        = lora,
            lora_dir    = lora_dir,
            lora_scale  = lora_scale,
            no_injection= no_injection,
        )

        # 5) 调用 DitailDemo
        ditail = DitailDemo(args)
        out_path = ditail.run_ditail_comfy()

        # 6) 读取结果 → Tensor 返回
        out_pil    = Image.open(out_path).convert("RGB")
        out_tensor = DitailStyleTransfer.pil_to_tensor(out_pil, device=device_str)

        return (out_tensor,)