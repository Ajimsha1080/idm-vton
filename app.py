from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from PIL import Image
import os
import uvicorn

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import build_pipeline_from_ckpt

# Path to trained model checkpoint (change if needed)
MODEL_CKPT_PATH = os.environ.get(
    "MODEL_CKPT_PATH",
    "./ckpt"   # default local ckpt folder
)

app = FastAPI()
pipeline = None


def load_pipeline(ckpt_path=None, device="cuda"):
    global pipeline
    if pipeline is not None:
        return pipeline

    ckpt = ckpt_path or MODEL_CKPT_PATH

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt}")

    pipeline = build_pipeline_from_ckpt(ckpt, device=device)
    return pipeline


@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    cloth: UploadFile = File(...),
    steps: int = 20,
    guidance: float = 2.0
):
    try:
        person_img = Image.open(BytesIO(await person.read())).convert("RGB")
        cloth_img = Image.open(BytesIO(await cloth.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input images: {e}")

    try:
        pipe = load_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    output = pipe(
        person_img,
        cloth_img,
        num_inference_steps=steps,
        guidance_scale=guidance
    )

    if isinstance(output, (list, tuple)):
        result_img = output[0]
    else:
        result_img = output

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
