from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from dotenv import load_dotenv

@dataclass(frozen=True)
class Config:
    coco_root: Path
    dataset_dir: Path
    model_dir: Path
    results_dir: Path
    categories: List[str]
    max_per_class: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    backbone: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    freeze_backbone: bool
    amp: bool
    device: str
    model_path: str
    topk: int

def _get_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device

def load_config() -> Config:
    load_dotenv(override=False)
    coco_root = Path(os.getenv("COCO_ROOT", "./coco"))
    dataset_dir = Path(os.getenv("DATASET_DIR", "./dataset_coco4cls"))
    model_dir = Path(os.getenv("MODEL_DIR", "./models"))
    results_dir = Path(os.getenv("RESULTS_DIR", "./results"))
    categories = [x.strip() for x in os.getenv("CATEGORIES", "cat,dog,car,bicycle").split(",") if x.strip()]
    max_per_class = int(os.getenv("MAX_PER_CLASS", "5000"))
    train_ratio = float(os.getenv("TRAIN_RATIO", "0.8"))
    val_ratio = float(os.getenv("VAL_RATIO", "0.1"))
    test_ratio = float(os.getenv("TEST_RATIO", "0.1"))
    seed = int(os.getenv("SEED", "42"))
    backbone = os.getenv("BACKBONE", "efficientnet_b0")
    image_size = int(os.getenv("IMAGE_SIZE", "224"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    epochs = int(os.getenv("EPOCHS", "10"))
    lr = float(os.getenv("LR", "0.0003"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "0.0001"))
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    freeze_backbone = os.getenv("FREEZE_BACKBONE", "0") == "1"
    amp = os.getenv("AMP", "1") == "1"
    device = _get_device(os.getenv("DEVICE", "auto"))
    model_path = os.getenv("MODEL_PATH", str(model_dir / "best.pt"))
    topk = int(os.getenv("TOPK", "4"))
    return Config(
        coco_root=coco_root,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        results_dir=results_dir,
        categories=categories,
        max_per_class=max_per_class,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        backbone=backbone,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        freeze_backbone=freeze_backbone,
        amp=amp,
        device=device,
        model_path=model_path,
        topk=topk,
    )
