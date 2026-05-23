"""Strategy expression-tree — the Python mirror of math-engine-cpp's AST.

A `Node` is one of:
  - `Var(name)`           — references a feature by canonical name
  - `Const(value)`        — float literal
  - `BinOp(op, a, b)`     — binary operator: add/sub/mul/div/min/max
  - `UnOp(op, a)`         — unary operator: tanh/abs/sign/neg

Trees are immutable (frozen dataclasses). Mutation operators construct new
trees; identical trees compare equal and hash to the same value.

The JSON serialization round-trips byte-for-identical with what the C++
dyn-eval expects (see math-engine-cpp/src/eval_dyn.cpp).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union


BIN_OPS = ("add", "sub", "mul", "div", "min", "max")
UN_OPS  = ("tanh", "abs", "sign", "neg")


@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Const:
    value: float

@dataclass(frozen=True)
class BinOp:
    op: str
    a: "Node"
    b: "Node"

@dataclass(frozen=True)
class UnOp:
    op: str
    a: "Node"


Node = Union[Var, Const, BinOp, UnOp]


# ---------------- JSON serialization ----------------

def to_dict(node: Node) -> dict:
    if isinstance(node, Var):
        return {"var": node.name}
    if isinstance(node, Const):
        return {"const": float(node.value)}
    if isinstance(node, BinOp):
        return {"op": node.op, "args": [to_dict(node.a), to_dict(node.b)]}
    if isinstance(node, UnOp):
        return {"op": node.op, "args": [to_dict(node.a)]}
    raise TypeError(f"unknown node type: {type(node).__name__}")


def from_dict(d: dict) -> Node:
    if "var" in d:
        return Var(str(d["var"]))
    if "const" in d:
        return Const(float(d["const"]))
    if "op" in d:
        op = str(d["op"])
        args = [from_dict(c) for c in d.get("args", [])]
        if op in BIN_OPS:
            if len(args) != 2:
                raise ValueError(f"binop {op!r} needs 2 args, got {len(args)}")
            return BinOp(op, args[0], args[1])
        if op in UN_OPS:
            if len(args) != 1:
                raise ValueError(f"unop {op!r} needs 1 arg, got {len(args)}")
            return UnOp(op, args[0])
        raise ValueError(f"unknown op: {op!r}")
    raise ValueError(f"malformed tree node: {d!r}")


# ---------------- Structural queries ----------------

def depth(node: Node) -> int:
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, UnOp):
        return 1 + depth(node.a)
    if isinstance(node, BinOp):
        return 1 + max(depth(node.a), depth(node.b))
    raise TypeError(type(node))


def complexity(node: Node) -> int:
    """Total node count (the PySR-style complexity measure)."""
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, UnOp):
        return 1 + complexity(node.a)
    if isinstance(node, BinOp):
        return 1 + complexity(node.a) + complexity(node.b)
    raise TypeError(type(node))


def used_features(node: Node) -> set[str]:
    out: set[str] = set()
    def visit(n: Node) -> None:
        if isinstance(n, Var):
            out.add(n.name)
        elif isinstance(n, UnOp):
            visit(n.a)
        elif isinstance(n, BinOp):
            visit(n.a); visit(n.b)
    visit(node)
    return out


def iter_subtrees(node: Node) -> Iterator[Node]:
    """Pre-order traversal yielding every subtree (including the root)."""
    yield node
    if isinstance(node, UnOp):
        yield from iter_subtrees(node.a)
    elif isinstance(node, BinOp):
        yield from iter_subtrees(node.a)
        yield from iter_subtrees(node.b)


def pretty(node: Node) -> str:
    """Human-readable infix string (best-effort, parens-heavy)."""
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Const):
        v = node.value
        return f"{v:.6g}"
    if isinstance(node, UnOp):
        return f"{node.op}({pretty(node.a)})"
    if isinstance(node, BinOp):
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(node.op)
        if sym:
            return f"({pretty(node.a)} {sym} {pretty(node.b)})"
        return f"{node.op}({pretty(node.a)}, {pretty(node.b)})"
    raise TypeError(type(node))


# ---------------- Numeric evaluation (cross-check helper) ----------------

_BIN_FNS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: 0.0 if b == 0.0 else a / b,
    "min": min,
    "max": max,
}
_UN_FNS = {
    "tanh": math.tanh,
    "abs":  abs,
    "sign": lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
    "neg":  lambda x: -x,
}

def evaluate(node: Node, features: dict[str, float]) -> float:
    if isinstance(node, Var):
        if node.name not in features:
            raise KeyError(f"unknown feature: {node.name}")
        return float(features[node.name])
    if isinstance(node, Const):
        return float(node.value)
    if isinstance(node, BinOp):
        return _BIN_FNS[node.op](evaluate(node.a, features),
                                 evaluate(node.b, features))
    if isinstance(node, UnOp):
        return _UN_FNS[node.op](evaluate(node.a, features))
    raise TypeError(type(node))


# ---------------- Spec wrapper (tree + metadata + JSON file IO) ----------------

@dataclass
class StrategyTreeSpec:
    """A symbolic strategy: tree + metadata. Mirrors the JSON file on disk."""
    id: str
    tree: Node
    output_scale: float = 1.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "expression_tree": to_dict(self.tree),
            "output_scale": self.output_scale,
        }
        # Round-trip preserve any extra fields (provenance, engine, etc.)
        for k, v in self.meta.items():
            if k not in ("id", "expression_tree", "output_scale"):
                d.setdefault(k, v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyTreeSpec":
        if "expression_tree" not in d:
            raise ValueError("strategy dict missing 'expression_tree'")
        meta = {k: v for k, v in d.items()
                if k not in ("id", "expression_tree", "output_scale")}
        return cls(
            id=str(d.get("id", "anonymous")),
            tree=from_dict(d["expression_tree"]),
            output_scale=float(d.get("output_scale", 1.0)),
            meta=meta,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StrategyTreeSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "BIN_OPS", "UN_OPS",
    "Var", "Const", "BinOp", "UnOp", "Node",
    "to_dict", "from_dict",
    "depth", "complexity", "used_features", "iter_subtrees", "pretty",
    "evaluate",
    "StrategyTreeSpec",
]
