import os
import sys
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import modal

app = modal.App("idm-vton-api")

# ---------------------------
# BUILD MODAL IMAGE
# ---------------------------
image = (
     modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("python-multipart")                 # REQUIRED for form uploads
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("pillow")
    .copy_local_dir(".", "/root/IDM-VTON")
    .pip_install_from_requirements("/root/IDM-VTON/requirements.txt")
    .image_id("force-rebuild-3")         
)

fastapi_app = FastAPI()
pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    sys.path.insert(0, "/root/IDM-VTON")

    from inference import build_pipeline_from_ckpt

    ckpt_folder = "/root/IDM-VTON/ckpt"
    if not os.path.exists(ckpt_folder):
        raise RuntimeError("Checkpoint folder not found")

    pipeline = build_pipeline_from_ckpt(ckpt_folder, device="cuda")
    return pipeline


@fastapi_app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    try:
        p_img = Image.open(BytesIO(await person.read())).convert("RGB")
        c_img = Image.open(BytesIO(await cloth.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    pipe = load_pipeline()
    result = pipe(p_img, c_img)

    if isinstance(result, (list, tuple)):
        result = result[0]

    buf = BytesIO()
    result.save(buf, "PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.function(image=image, gpu="A10G", timeout=1800)
@modal.asgi_app()
def api():
    return fastapi_app
