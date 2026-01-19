from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from coco4cls.utils import ensure_dir


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    label: int
    label_name: str
    split: str


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, preprocess: transforms.Compose):
        df = pd.read_csv(csv_path)
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain columns: image_path, label")
        self.image_paths = df["image_path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        y = self.labels[idx]
        img = Image.open(p).convert("RGB")
        x = self.preprocess(img)
        return x, torch.tensor(y, dtype=torch.long)


def _resolve_coco_paths(coco_root: Path) -> Dict[str, Path]:
    images_train = coco_root / "images" / "train2017"
    images_val = coco_root / "images" / "val2017"
    ann_dir = coco_root / "annotations"
    ann_train = ann_dir / "instances_train2017.json"
    ann_val = ann_dir / "instances_val2017.json"
    return {
        "images_train": images_train,
        "images_val": images_val,
        "ann_train": ann_train,
        "ann_val": ann_val,
    }


def _split_list(items: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def prepare_coco4cls_manifests(
    coco_root: Path,
    dataset_dir: Path,
    categories: List[str],
    max_per_class: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    strict_single_selected: bool = True,
) -> Dict[str, Path]:
    ensure_dir(dataset_dir)

    paths = _resolve_coco_paths(coco_root)
    ann_train = paths["ann_train"]
    ann_val = paths["ann_val"]
    img_train_dir = paths["images_train"]
    img_val_dir = paths["images_val"]

    if not ann_train.exists() or not ann_val.exists():
        raise FileNotFoundError("COCO annotations not found. Run scripts/download_coco2017.py first.")

    from pycocotools.coco import COCO

    coco_train = COCO(str(ann_train))
    coco_val = COCO(str(ann_val))

    def build_image_to_present(coco: COCO) -> Dict[int, List[int]]:
        name_to_catid = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
        selected_cat_ids = []
        for n in categories:
            if n not in name_to_catid:
                raise ValueError(f"Category '{n}' not found in COCO categories")
            selected_cat_ids.append(name_to_catid[n])

        img_to_present: Dict[int, List[int]] = {}
        for local_label, cat_id in enumerate(selected_cat_ids):
            img_ids = coco.getImgIds(catIds=[cat_id])
            for img_id in img_ids:
                img_to_present.setdefault(img_id, []).append(local_label)
        return img_to_present

    train_map = build_image_to_present(coco_train)
    val_map = build_image_to_present(coco_val)

    def collect_rows(coco: COCO, img_dir: Path, img_to_present: Dict[int, List[int]]) -> List[ManifestRow]:
        rows: List[ManifestRow] = []
        for img_id, present_labels in img_to_present.items():
            uniq = sorted(set(present_labels))
            if strict_single_selected and len(uniq) != 1:
                continue
            if len(uniq) == 0:
                continue
            label = uniq[0]
            img_info = coco.loadImgs([img_id])[0]
            image_path = str((img_dir / img_info["file_name"]).resolve())
            rows.append(ManifestRow(image_path=image_path, label=int(label), label_name=categories[int(label)], split=""))
        return rows

    all_rows = collect_rows(coco_train, img_train_dir, train_map) + collect_rows(coco_val, img_val_dir, val_map)

    by_class: Dict[int, List[str]] = {i: [] for i in range(len(categories))}
    for r in all_rows:
        by_class[r.label].append(r.image_path)

    rng = random.Random(seed)
    train_rows: List[ManifestRow] = []
    val_rows: List[ManifestRow] = []
    test_rows: List[ManifestRow] = []

    for k in range(len(categories)):
        items = by_class[k][:]
        rng.shuffle(items)
        items = items[:max_per_class]
        tr, va, te = _split_list(items, train_ratio, val_ratio, test_ratio, seed + k)

        train_rows += [ManifestRow(p, k, categories[k], "train") for p in tr]
        val_rows += [ManifestRow(p, k, categories[k], "val") for p in va]
        test_rows += [ManifestRow(p, k, categories[k], "test") for p in te]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    def write_csv(rows: List[ManifestRow], out_path: Path) -> None:
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(out_path, index=False)

    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    test_csv = dataset_dir / "test.csv"

    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)
    write_csv(test_rows, test_csv)

    return {"train": train_csv, "val": val_csv, "test": test_csv}
