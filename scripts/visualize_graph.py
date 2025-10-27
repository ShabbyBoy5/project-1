#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.viz.graphs import draw_edges_json_to_png


def main():
    ap = argparse.ArgumentParser(description="Visualize causal edges JSON to a PNG.")
    ap.add_argument("--edges", type=str, required=True, help="Path to edges JSON (e.g., graph_fused.json)")
    ap.add_argument("--columns", type=str, required=True, help="Path to columns.json")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")
    ap.add_argument("--title", type=str, default="Causal Graph (lagged)")
    ap.add_argument("--min_score", type=float, default=None)
    args = ap.parse_args()

    out = draw_edges_json_to_png(args.edges, args.columns, args.out, title=args.title, min_score=args.min_score)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
