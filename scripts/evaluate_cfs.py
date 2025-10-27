#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import cf_summary


def main():
    ap = argparse.ArgumentParser(description="Summarize counterfactual results JSON.")
    ap.add_argument("--cf_json", type=str, required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.cf_json).read_text())
    s = cf_summary(data)
    out = Path(args.cf_json).with_suffix(".summary.json")
    out.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
