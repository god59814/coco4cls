from __future__ import annotations
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from coco4cls.config import load_config
from coco4cls.model import build_model, build_preprocess, predict_image

app = FastAPI(title="coco4cls", version="1.0.0")

CFG = None
MODEL = None
PREPROCESS = None

@app.on_event("startup")
def _startup() -> None:
    global CFG, MODEL, PREPROCESS
    CFG = load_config()
    PREPROCESS = build_preprocess(CFG.image_size)

    model_path = Path(CFG.model_path)
    if not model_path.exists():
        raise RuntimeError(f"MODEL_PATH not found: {model_path}")

    MODEL = build_model(
        backbone=CFG.backbone,
        num_classes=len(CFG.categories),
        freeze_backbone=False,
        pretrained=False,
    )
    state = torch.load(model_path, map_location="cpu")
    MODEL.load_state_dict(state, strict=True)
    MODEL.eval()
    MODEL.to(CFG.device)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": str(CFG.device), "categories": CFG.categories, "model_path": CFG.model_path}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = predict_image(MODEL, img, PREPROCESS, CFG.device, CFG.categories, topk=CFG.topk)
    return JSONResponse(result)

@app.post("/predict_csv")
async def predict_csv(file: Optional[UploadFile] = File(None), csv_path: Optional[str] = None) -> JSONResponse:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if file is None and not csv_path:
        raise HTTPException(status_code=400, detail="Provide either file or csv_path")

    if file is not None:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        p = Path(csv_path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"csv_path not found: {p}")
        df = pd.read_csv(p)

    if "image_path" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'image_path' column")

    outputs: List[Dict[str, Any]] = []
    for p in df["image_path"].astype(str).tolist():
        img_path = Path(p)
        if not img_path.exists():
            outputs.append({"image_path": p, "error": "not found"})
            continue
        img = Image.open(img_path).convert("RGB")
        pred = predict_image(MODEL, img, PREPROCESS, CFG.device, CFG.categories, topk=CFG.topk)
        pred["image_path"] = p
        outputs.append(pred)

    return JSONResponse({"results": outputs})
