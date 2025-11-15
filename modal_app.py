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

app = modal.App("idm-vton-api")

# ---------------------------------------------------------
# GPU IMAGE WITH CORRECT DEPENDENCIES (NO GIT+)
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
        "torch",
        "torchvision",
    )
    # SAFE MIRROR LINKS (NO GITHUB DOWNLOAD)
    .pip_install("https://huggingface.co/yisol/diffusers/resolve/main/diffusers.zip")
    .pip_install("https://huggingface.co/tencent-ailab/IP-Adapter/resolve/main/ipadapter.zip")
)



fastapi_app = FastAPI()
pipeline = None


def clone_repo():
    repo_path = "/root/IDM-VTON"
    if not os.path.exists(repo_path):
        subprocess.check_call(["git", "clone", GIT_URL, repo_path])

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    return repo_path


def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    repo_path = clone_repo()

    from inference import build_pipeline_from_ckpt
    ckpt_path = os.path.join(repo_path, "ckpt")
    pipeline = build_pipeline_from_ckpt(ckpt_path, device="cuda")

    return pipeline


@fastapi_app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    try:
        person_img = Image.open(BytesIO(await person.read())).convert("RGB")
        cloth_img = Image.open(BytesIO(await cloth.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    pipe = load_pipeline()
    output = pipe(person_img, cloth_img)

    buf = BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.function(
    image=base,
    gpu="A10G",
    timeout=600,
)
@modal.asgi_app()
def api():
    return fastapi_app

