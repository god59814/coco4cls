from __future__ import annotations

import argparse

from coco4cls.config import load_config
from coco4cls.eval import evaluate_main


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    out = evaluate_main(cfg, split=args.split)
    print(out)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str((_Path(__file__).resolve().parents[1] / "src").resolve()))
    main()
