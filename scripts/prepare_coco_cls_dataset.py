from __future__ import annotations

import argparse
from pathlib import Path

from coco4cls.config import load_config
from coco4cls.data import prepare_coco4cls_manifests


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-single-selected", type=int, default=1)
    ap.add_argument("--max-per-class", type=int, default=cfg.max_per_class)
    args = ap.parse_args()

    out = prepare_coco4cls_manifests(
        coco_root=cfg.coco_root,
        dataset_dir=cfg.dataset_dir,
        categories=cfg.categories,
        max_per_class=args.max_per_class,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        strict_single_selected=(args.strict_single_selected == 1),
    )

    print("Dataset prepared:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str((_Path(__file__).resolve().parents[1] / "src").resolve()))
    main()
