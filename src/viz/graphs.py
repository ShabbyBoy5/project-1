from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import networkx as nx


def load_edges(edges_path: str | Path) -> List[dict]:
    data = json.loads(Path(edges_path).read_text())
    # Support either list of dicts or list of objects with attributes
    edges = []
    for e in data:
        if isinstance(e, dict):
            edges.append(e)
        else:
            edges.append({k: getattr(e, k) for k in ["src", "dst", "lag", "score"] if hasattr(e, k)})
    return edges


def load_columns(columns_path: str | Path) -> Sequence[str]:
    return json.loads(Path(columns_path).read_text())


def draw_edges_json_to_png(
    edges_path: str | Path,
    columns_path: str | Path,
    out_png: str | Path,
    title: str = "Causal Graph (lagged)",
    min_score: float | None = None,
) -> Path:
    edges = load_edges(edges_path)
    cols = load_columns(columns_path)
    G = nx.DiGraph()
    for i, name in enumerate(cols):
        G.add_node(i, label=name)
    for e in edges:
        s = float(e.get("score", 0.0))
        if min_score is not None and s < min_score:
            continue
        src, dst, lag = int(e["src"]), int(e["dst"]), int(e["lag"])
        label = f"lag={lag}\n{round(s,3)}"
        G.add_edge(src, dst, weight=s, label=label)

    pos = nx.spring_layout(G, seed=42, k=1.2/(len(G.nodes())+1))
    plt.figure(figsize=(10, 7))
    node_labels = {i: G.nodes[i]["label"] for i in G.nodes}
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#e0f3ff", edgecolors="#1f77b4")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    widths = [1 + 2 * G[u][v].get("weight", 0.0) for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, arrows=True, width=widths, edge_color="#444")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title(title)
    plt.axis("off")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png
