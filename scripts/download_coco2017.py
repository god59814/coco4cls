from __future__ import annotations
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from coco4cls.config import load_config
from coco4cls.utils import ensure_dir

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def _download(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists():
        return
    print(f"Downloading: {url}")
    urlretrieve(url, str(out_path))

def _unzip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

def main() -> None:
    cfg = load_config()
    coco_root = cfg.coco_root.resolve()
    ensure_dir(coco_root)

    zips_dir = coco_root / "_zips"
    ensure_dir(zips_dir)

    train_zip = zips_dir / "train2017.zip"
    val_zip = zips_dir / "val2017.zip"
    ann_zip = zips_dir / "annotations_trainval2017.zip"

    _download(COCO_URLS["train2017"], train_zip)
    _download(COCO_URLS["val2017"], val_zip)
    _download(COCO_URLS["annotations"], ann_zip)

    images_dir = coco_root / "images"
    ann_dir = coco_root / "annotations"
    ensure_dir(images_dir)
    ensure_dir(ann_dir)

    if not (images_dir / "train2017").exists():
        print("Unzipping train2017...")
        _unzip(train_zip, images_dir)
    if not (images_dir / "val2017").exists():
        print("Unzipping val2017...")
        _unzip(val_zip, images_dir)
    if not (ann_dir / "instances_train2017.json").exists():
        print("Unzipping annotations...")
        _unzip(ann_zip, coco_root)

    print("COCO 2017 download done.")
    print(f"COCO_ROOT: {coco_root}")

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.append(str((_Path(__file__).resolve().parents[1] / "src").resolve()))
    main()
