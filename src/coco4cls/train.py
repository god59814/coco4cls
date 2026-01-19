from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco4cls.config import Config
from coco4cls.data import CsvImageDataset
from coco4cls.model import build_model, build_preprocess
from coco4cls.utils import ensure_dir, set_seed, append_jsonl


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def train_main(cfg: Config) -> Dict[str, Any]:
    set_seed(cfg.seed)
    ensure_dir(cfg.model_dir)

    train_csv = cfg.dataset_dir / "train.csv"
    val_csv = cfg.dataset_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("train.csv/val.csv not found. Run scripts/prepare_coco_cls_dataset.py first.")

    preprocess = build_preprocess(cfg.image_size)
    ds_train = CsvImageDataset(train_csv, preprocess=preprocess)
    ds_val = CsvImageDataset(val_csv, preprocess=preprocess)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = build_model(
        backbone=cfg.backbone,
        num_classes=len(cfg.categories),
        freeze_backbone=cfg.freeze_backbone,
        pretrained=True,
    ).to(cfg.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device == "cuda"))

    best_val_acc = -1.0
    best_path = cfg.model_dir / "best.pt"
    last_path = cfg.model_dir / "last.pt"
    log_path = cfg.model_dir / "train_log.jsonl"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        for x, y in tqdm(dl_train, desc=f"train epoch {epoch}", leave=False):
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device == "cuda")):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            bsz = y.size(0)
            train_loss += float(loss.item()) * bsz
            train_acc += _accuracy(logits.detach(), y) * bsz
            n_train += bsz

        sched.step()
        train_loss /= max(n_train, 1)
        train_acc /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.inference_mode():
            for x, y in tqdm(dl_val, desc=f"val epoch {epoch}", leave=False):
                x = x.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)

                bsz = y.size(0)
                val_loss += float(loss.item()) * bsz
                val_acc += _accuracy(logits, y) * bsz
                n_val += bsz

        val_loss /= max(n_val, 1)
        val_acc /= max(n_val, 1)

        torch.save(model.state_dict(), last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        append_jsonl(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "lr": float(optim.param_groups[0]["lr"]),
                "time_sec": float(time.time() - t0),
            },
        )

        print(f"[epoch {epoch}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}")

    return {"best_val_acc": best_val_acc, "best_path": str(best_path), "last_path": str(last_path)}
