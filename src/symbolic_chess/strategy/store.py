"""Strategy spec storage.

A 'strategy' is a closed-form eval expression plus the metadata needed to
reproduce it: feature list, constants, material weights, and a pointer to the
UCI engine binary that implements it. The current cx=13 spec was distilled
from Stockfish (iter-0); future self-play iterations save new specs here in
the same shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class StrategySpec:
    id: str
    version: int
    provenance: dict[str, Any]
    engine: dict[str, Any]
    features: list[str]
    material_unit_weights: dict[str, float]
    constants: dict[str, float]
    expression: str
    safe_div: bool = True
    output_units: str = "centipawns_white_perspective"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_strategy(path: str | Path) -> StrategySpec:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return StrategySpec(
        id=d["id"],
        version=int(d["version"]),
        provenance=d.get("provenance", {}),
        engine=d.get("engine", {}),
        features=list(d["features"]),
        material_unit_weights=dict(d["material_unit_weights"]),
        constants=dict(d["constants"]),
        expression=d["expression"],
        safe_div=bool(d.get("safe_div", True)),
        output_units=d.get("output_units", "centipawns_white_perspective"),
    )


def save_strategy(spec: StrategySpec, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)
        f.write("\n")
    return p
