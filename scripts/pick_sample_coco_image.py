from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from pycocotools.coco import COCO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", type=str, default="coco")
    ap.add_argument("--category", type=str, default="cat", choices=["cat", "dog", "car", "bicycle"])
    ap.add_argument("--split", type=str, default="val2017", choices=["train2017", "val2017"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    ann = coco_root / "annotations" / f"instances_{args.split}.json"
    img_dir = coco_root / "images" / args.split

    if not ann.exists():
        raise FileNotFoundError(f"COCO annotation not found: {ann}")
    if not img_dir.exists():
        raise FileNotFoundError(f"COCO images not found: {img_dir}")

    coco = COCO(str(ann))
    cats = coco.loadCats(coco.getCatIds(catNms=[args.category]))
    if not cats:
        raise RuntimeError(f"Category not found in COCO: {args.category}")
    cat_id = cats[0]["id"]

    img_ids = coco.getImgIds(catIds=[cat_id])
    if not img_ids:
        raise RuntimeError(f"No images found for category: {args.category}")

    random.seed(args.seed)
    img_id = random.choice(img_ids)
    info = coco.loadImgs([img_id])[0]
    src = img_dir / info["file_name"]

    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "sample.jpg"

    shutil.copy2(src, dst)

    meta_dir = Path("docs")
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_dir / "sample_image_source.md"
    meta.write_text(
        f"# Sample Image Source\n\n"
        f"- Source dataset: MS COCO 2017\n"
        f"- Split: {args.split}\n"
        f"- Category (query): {args.category}\n"
        f"- COCO image_id: {img_id}\n"
        f"- Original file_name: {info['file_name']}\n"
        f"- Copied to: data/sample.jpg\n",
        encoding="utf-8",
    )

    print(f"OK: copied {src} -> {dst}")
    print(f"Saved source info: {meta}")


if __name__ == "__main__":
    main()
