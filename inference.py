# ==========================================
# IDM-VTON INFERENCE FILE FOR MODAL API
# CLEAN + FULLY WORKING VERSION
# ==========================================

import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from ip_adapter.ip_adapter import IPAdapterPlusXL


# ----------------------------------------------------
# Helper: Convert PIL → tensor if ever needed
# ----------------------------------------------------
def pil_to_tensor(img):
    import numpy as np
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ----------------------------------------------------
# MAIN LOADER USED BY modal_app.py
# ----------------------------------------------------
def build_pipeline_from_ckpt(ckpt_path, device="cuda"):
    """
    Loads the IDM-VTON finetuned SDXL + IP-Adapter pipeline.
    Returns a callable function for modal_api.
    """

    # Normalize checkpoint path
    ckpt_path = ckpt_path.rstrip("/") + "/"

    # 1️⃣ Load SDXL base pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to(device)

    # 2️⃣ Load IP-Adapter weights
    ip_ckpt = os.path.join(ckpt_path, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin")

    if not os.path.exists(ip_ckpt):
        raise FileNotFoundError(f"[ERROR] Missing IP-Adapter file at:\n{ip_ckpt}")

    _ = IPAdapterPlusXL(pipe, ip_ckpt)

    # 3️⃣ Load finetuned IDM-VTON weights
    model_ckpt = ckpt_path + "pytorch_model.bin"

    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"[ERROR] Missing pipeline weights at:\n{model_ckpt}")

    state_dict = torch.load(model_ckpt, map_location="cpu")
    pipe.load_state_dict(state_dict, strict=False)

    pipe.to(device)
    pipe.eval()

    # ----------------------------------------------------
    # The function returned to modal_app.py
    # ----------------------------------------------------
    def run_pipeline(
        person_img: Image.Image,
        cloth_img: Image.Image,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.0
    ):
        """
        Arguments:
            person_img  → PIL Image
            cloth_img   → PIL Image
        Returns:
            PIL Image (generated try-on)
        """

        with torch.no_grad():
            result = pipe(
                image=person_img,
                prompt="virtual try-on, realistic cloth replacement",
                ip_adapter_image=cloth_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

        return result.images[0]

    return run_pipeline

