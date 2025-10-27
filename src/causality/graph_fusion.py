from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class EdgeKey:
    src: int
    dst: int
    lag: int


def fuse_union(
    pcmci_edges: Iterable[dict] | Iterable[object],
    granger_edges: Iterable[dict] | Iterable[object],
    weight_mode: str = "avg",
) -> List[dict]:
    """
    Fuse edges by union. Edges represented as dicts: {src,dst,lag,score/weight,pval}.
    weight_mode: avg|max prioritizes combining scores.
    """
    acc: Dict[EdgeKey, Dict] = {}

    def add_edge(e: dict, kind: str):
        k = EdgeKey(int(e["src"]), int(e["dst"]), int(e["lag"]))
        if k not in acc:
            acc[k] = {"src": k.src, "dst": k.dst, "lag": k.lag, "from": set([kind])}
        acc[k][f"{kind}_score"] = float(e.get("score", e.get("weight", 0.0)))
        if "pval" in e:
            acc[k][f"{kind}_pval"] = float(e["pval"])
        acc[k]["from"].add(kind)

    for e in pcmci_edges:
        add_edge(e if isinstance(e, dict) else e.__dict__, "pcmci")
    for e in granger_edges:
        add_edge(e if isinstance(e, dict) else e.__dict__, "granger")

    fused: List[dict] = []
    for k, v in acc.items():
        s1 = v.get("pcmci_score")
        s2 = v.get("granger_score")
        if s1 is not None and s2 is not None:
            s = max(s1, s2) if weight_mode == "max" else (s1 + s2) / 2.0
        else:
            s = s1 if s1 is not None else s2 if s2 is not None else 0.0
        fused.append({
            "src": v["src"],
            "dst": v["dst"],
            "lag": v["lag"],
            "score": float(s),
            "from": sorted(list(v["from"]))
        })
    return fused


def fuse_intersection(
    pcmci_edges: Iterable[dict] | Iterable[object],
    granger_edges: Iterable[dict] | Iterable[object],
    weight_mode: str = "avg",
) -> List[dict]:
    """
    Keep only edges present in both methods.
    """
    def key(e) -> EdgeKey:
        d = e if isinstance(e, dict) else e.__dict__
        return EdgeKey(int(d["src"]), int(d["dst"]), int(d["lag"]))

    pcmci_map = {key(e): (e if isinstance(e, dict) else e.__dict__) for e in pcmci_edges}
    granger_map = {key(e): (e if isinstance(e, dict) else e.__dict__) for e in granger_edges}
    common = []
    for k, e1 in pcmci_map.items():
        if k in granger_map:
            e2 = granger_map[k]
            s1 = float(e1.get("score", e1.get("weight", 0.0)))
            s2 = float(e2.get("score", e2.get("weight", 0.0)))
            s = max(s1, s2) if weight_mode == "max" else (s1 + s2) / 2.0
            common.append({"src": k.src, "dst": k.dst, "lag": k.lag, "score": s, "from": ["pcmci", "granger"]})
    return common
