from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from coco4cls.config import Config
from coco4cls.data import CsvImageDataset
from coco4cls.model import build_model, build_preprocess
from coco4cls.utils import ensure_dir, plot_confusion_matrix, save_json


def _one_vs_rest_cm(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    yt = (y_true == k).astype(int)
    yp = (y_pred == k).astype(int)
    return confusion_matrix(yt, yp, labels=[0, 1])


def _ovr_accuracy(cm2x2: np.ndarray) -> float:
    tn, fp, fn, tp = cm2x2.ravel()
    denom = tn + fp + fn + tp
    return float((tn + tp) / denom) if denom > 0 else 0.0


@torch.inference_mode()
def evaluate_main(cfg: Config, split: str = "test") -> Dict[str, Any]:
    ensure_dir(cfg.results_dir)

    csv_path = cfg.dataset_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{split}.csv not found. Run scripts/prepare_coco_cls_dataset.py first.")

    preprocess = build_preprocess(cfg.image_size)
    ds = CsvImageDataset(csv_path, preprocess=preprocess)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        model_path = cfg.model_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path} or {cfg.model_dir / 'best.pt'}")

    model = build_model(
        backbone=cfg.backbone,
        num_classes=len(cfg.categories),
        freeze_backbone=False,
        pretrained=False,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(cfg.device)

    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in dl:
        x = x.to(cfg.device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    cm_4x4 = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(cfg.categories))))

    out: Dict[str, Any] = {
        "split": split,
        "model_path": str(model_path),
        "num_samples": int(len(y_true_arr)),
        "overall_accuracy": float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0,
        "per_class": {},
    }

    cm4_png = cfg.results_dir / "confusion_matrix_4x4.png"
    cm4_csv = cfg.results_dir / "confusion_matrix_4x4.csv"
    pd.DataFrame(cm_4x4, index=cfg.categories, columns=cfg.categories).to_csv(cm4_csv)
    plot_confusion_matrix(cm_4x4, cfg.categories, title="Confusion Matrix (4x4)", out_path=cm4_png)

    for k, name in enumerate(cfg.categories):
        cm2 = _one_vs_rest_cm(y_true_arr, y_pred_arr, k)
        acc_k = _ovr_accuracy(cm2)

        out["per_class"][name] = {
            "class_id": int(k),
            "one_vs_rest_accuracy": float(acc_k),
            "cm_2x2": cm2.tolist(),
        }

        cm2_png = cfg.results_dir / f"class_{name}_cm_2x2.png"
        cm2_csv = cfg.results_dir / f"class_{name}_cm_2x2.csv"
        pd.DataFrame(cm2, index=["neg", "pos"], columns=["neg", "pos"]).to_csv(cm2_csv)
        plot_confusion_matrix(cm2, ["neg", "pos"], title=f"One-vs-Rest CM (2x2) - {name}", out_path=cm2_png)

    metrics_path = cfg.results_dir / "metrics.json"
    save_json(metrics_path, out)

    return out
