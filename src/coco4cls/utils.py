from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def plot_confusion_matrix(cm: np.ndarray, labels, title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    cm_int = cm.astype(int)
    thresh = cm_int.max() / 2.0 if cm_int.size else 0.0
    for i in range(cm_int.shape[0]):
        for j in range(cm_int.shape[1]):
            plt.text(j, i, str(cm_int[i, j]), ha="center", va="center", color="white" if cm_int[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
