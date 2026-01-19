from __future__ import annotations

from coco4cls.config import load_config
from coco4cls.train import train_main


def main() -> None:
    cfg = load_config()
    out = train_main(cfg)
    print(out)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str((_Path(__file__).resolve().parents[1] / "src").resolve()))
    main()
