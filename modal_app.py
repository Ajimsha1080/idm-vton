# modal_app.py
import modal
import os
import sys
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

GIT_URL = "https://github.com/Ajimsha1080/idm-vton.git"

# NEW APP NAME (forces Modal to rebuild)
app = modal.App("idm-vton-api-v5")


# ---------------------------------------------------------
# GPU IMAGE WITH SAFE DEPENDENCIES (NO GIT+)
# ---------------------------------------------------------
base = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "fastapi",
        "uvicorn",
        "starlette",
        "python-multipart",
        "pillow",
        "accelerate",
        "transformers",
        "einops",
        "timm",
        "opencv-python-headless",
    )
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    # SAFE TAR.GZ ARCHIVES (no github authentication needed)
    .pip_install("https://github.com/yisol/diffusers/archive/refs/heads/main.tar.gz")
    .pip_install("https://github.com/tencent-ailab/IP-Adapter/archive/refs/heads/main.tar.gz")
)


fastapi_app = FastAPI()
pipeline = None


# ---------------------------------------------------------
# Clone IDM-VTON repository
# ---------------------------------------------------------
def clone_repo():
    repo_path = "/root/IDM-VTON"

    if not os.path.exists(repo_path):
        subprocess.check_call(["git", "clone", GIT_URL, repo_path])

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    return repo_path


# ---------------------------------------------------------
# Load the IDM-VTON pipeline
# ---------------------------------------------------------
def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    repo_path = clone_repo()

    from inference import build_pipeline_from_ckpt
    ckpt_path = os.path.join(repo_path, "ckpt")

    pipeline = build_pipeline_from_ckpt(ckpt_path, device="cuda")
    return pipeline


# ---------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------
@fastapi_app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    try:
        person_img = Image.open(BytesIO(await person.read())).convert("RGB")
        cloth_img = Image.open(BytesIO(await cloth.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pipe = load_pipeline()
    output = pipe(person_img, cloth_img)

    buffer = BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# ---------------------------------------------------------
# DEPLOY APP — FORCE NEW IMAGE BUILD
# ---------------------------------------------------------
@app.function(
    image=base.new(),   # <<< THIS FORCES MODAL TO REBUILD IMAGE
    gpu="A10G",
    timeout=600,
)
@modal.asgi_app()
def api():
    return fastapi_app
