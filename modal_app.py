# modal_app.py
import os
import sys
import subprocess
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import modal

# ------------------------------
# Your GitHub repo URL
# ------------------------------
GIT_URL = "https://github.com/Ajimsha1080/idm-vton.git"

app = modal.App("idm-vton-api")

# ------------------------------
# Build image with all packages
# ------------------------------
image = (
     modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("python-multipart")
    .pip_install("pillow")
    .run_commands(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    )
)

# FastAPI instance
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
    if pipeline:
        return pipeline

    repo_path = clone_repo()

    from inference import build_pipeline_from_ckpt

    ckpt_path = os.path.join(repo_path, "ckpt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError("Missing ckpt folder inside GitHub repo.")

    pipeline = build_pipeline_from_ckpt(ckpt_path, device="cuda")
    return pipeline


@fastapi_app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    try:
        p_img = Image.open(BytesIO(await person.read())).convert("RGB")
        c_img = Image.open(BytesIO(await cloth.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    pipe = load_pipeline()
    output = pipe(p_img, c_img)

    if isinstance(output, (list, tuple)):
        output = output[0]

    buffer = BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.function(image=image, gpu="A10G", timeout=1500)
@modal.asgi_app()
def api():
    return fastapi_app
