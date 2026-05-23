"""Mutation operators for symbolic-AlphaZero population evolution.

Six operators + subtree crossover. Each takes a `Node` (and an `rng`) and
returns a new `Node`. Operators may fail to produce a valid tree (e.g. would
exceed depth/complexity caps); the orchestrator should retry with a different
operator or different mutation point.

Validators (validate_tree) catch:
  - depth > MAX_DEPTH
  - complexity > MAX_COMPLEXITY
  - unknown feature variables
  - constant magnitude blow-up
The population only accepts mutations that pass validate_tree.

Constants of the trade:
  - Mutations construct new immutable trees (Node is frozen).
  - Probability weights for each operator are exposed in OP_WEIGHTS so the
    orchestrator can tune them per generation if needed.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from symbolic_chess.strategy.tree import (
    BIN_OPS, UN_OPS, BinOp, Const, Node, UnOp, Var,
    complexity, depth, iter_subtrees, used_features,
)


# ---------------- Constraints ----------------

MAX_DEPTH = 8
MAX_COMPLEXITY = 40
MAX_CONST_MAGNITUDE = 1e4   # reject jitters that blow constants up


def validate_tree(node: Node, feature_names: set[str]) -> Optional[str]:
    """Return None if valid, else a short error string explaining why not."""
    if depth(node) > MAX_DEPTH:
        return f"depth {depth(node)} > MAX_DEPTH {MAX_DEPTH}"
    if complexity(node) > MAX_COMPLEXITY:
        return f"complexity {complexity(node)} > MAX_COMPLEXITY {MAX_COMPLEXITY}"
    unknown = used_features(node) - feature_names
    if unknown:
        return f"unknown features: {sorted(unknown)}"
    for sub in iter_subtrees(node):
        if isinstance(sub, Const) and (abs(sub.value) > MAX_CONST_MAGNITUDE
                                       or not math.isfinite(sub.value)):
            return f"constant magnitude blow-up: {sub.value}"
    return None


# ---------------- Random tree builders (small subtrees) ----------------

def _random_const(rng: random.Random) -> Const:
    """A small float in a useful range — log-uniform on (0.01, 10), random sign."""
    sign = rng.choice([-1.0, 1.0])
    mag = math.exp(rng.uniform(math.log(0.01), math.log(10.0)))
    # Round to 6 sig figs so saved specs stay readable
    return Const(round(sign * mag, 6))


def _random_var(rng: random.Random, feature_names: list[str]) -> Var:
    return Var(rng.choice(feature_names))


def random_subtree(
    rng: random.Random,
    feature_names: list[str],
    max_depth: int = 3,
) -> Node:
    """Build a small random subtree, used as fresh material for subtree_swap."""
    if max_depth <= 1:
        # leaf
        if rng.random() < 0.5:
            return _random_var(rng, feature_names)
        return _random_const(rng)
    # Internal node — 70% binop, 20% unop, 10% leaf
    r = rng.random()
    if r < 0.10:
        if rng.random() < 0.5:
            return _random_var(rng, feature_names)
        return _random_const(rng)
    if r < 0.30:
        op = rng.choice(UN_OPS)
        return UnOp(op, random_subtree(rng, feature_names, max_depth - 1))
    op = rng.choice(BIN_OPS)
    return BinOp(
        op,
        random_subtree(rng, feature_names, max_depth - 1),
        random_subtree(rng, feature_names, max_depth - 1),
    )


# ---------------- Tree-rewrite helpers ----------------

def _replace_at(root: Node, target_id: int, new_node: Node) -> Node:
    """Walk root pre-order; when we hit the Nth subtree (by id()), substitute it.

    Pre-order index, NOT identity-based. We return a freshly constructed root
    with that subtree swapped.
    """
    counter = [0]
    def visit(n: Node) -> Node:
        idx = counter[0]
        counter[0] += 1
        if idx == target_id:
            return new_node
        if isinstance(n, (Var, Const)):
            return n
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.a))
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.a), visit(n.b))
        raise TypeError(type(n))
    return visit(root)


def _pick_subtree_index(rng: random.Random, root: Node) -> int:
    n = sum(1 for _ in iter_subtrees(root))
    return rng.randrange(n)


# ---------------- Individual mutation operators ----------------

def subtree_swap(
    tree: Node,
    rng: random.Random,
    feature_names: list[str],
    *,
    subtree_max_depth: int = 3,
) -> Node:
    """Replace a random subtree with a fresh random subtree."""
    idx = _pick_subtree_index(rng, tree)
    new_sub = random_subtree(rng, feature_names, max_depth=subtree_max_depth)
    return _replace_at(tree, idx, new_sub)


def subtree_crossover(
    parent_a: Node,
    parent_b: Node,
    rng: random.Random,
) -> Node:
    """Cut a subtree from B, splice into A at a random point."""
    b_subs = list(iter_subtrees(parent_b))
    donor = rng.choice(b_subs)
    idx = _pick_subtree_index(rng, parent_a)
    return _replace_at(parent_a, idx, donor)


def constant_jitter(
    tree: Node,
    rng: random.Random,
    *,
    sigma: float = 0.1,
) -> Node:
    """Multiply every Const in the tree by exp(N(0, sigma)). Lamarckian fine-tune."""
    def visit(n: Node) -> Node:
        if isinstance(n, Const):
            mult = math.exp(rng.gauss(0.0, sigma))
            return Const(round(n.value * mult, 6))
        if isinstance(n, Var):
            return n
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.a))
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.a), visit(n.b))
        raise TypeError(type(n))
    return visit(tree)


def term_insert(
    tree: Node,
    rng: random.Random,
    feature_names: list[str],
) -> Node:
    """Wrap root in `root +/- <random subtree>` — grows complexity by ~3-5."""
    op = rng.choice(["add", "sub"])
    addend = random_subtree(rng, feature_names, max_depth=2)
    return BinOp(op, tree, addend)


def term_delete(
    tree: Node,
    rng: random.Random,
) -> Node:
    """If a random subtree is Add/Sub, drop one of its operands. Otherwise no-op
    (returns the input). Use validate to check the result before keeping."""
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, BinOp) and s.op in ("add", "sub")
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    assert isinstance(sub, BinOp)
    replacement = sub.a if rng.random() < 0.5 else sub.b
    return _replace_at(tree, idx, replacement)


_OP_SWAP_GROUPS = [
    {"add", "sub"},
    {"mul", "div"},
    {"min", "max"},
    {"tanh", "sign"},
    {"abs", "neg"},
]

def op_swap(tree: Node, rng: random.Random) -> Node:
    """Swap an operator at a random internal node with a related one."""
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, (BinOp, UnOp))
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    # Find the group containing this op
    group = next((g for g in _OP_SWAP_GROUPS if sub.op in g), None)
    if group is None:
        return tree
    other_ops = [o for o in group if o != sub.op]
    if not other_ops:
        return tree
    new_op = rng.choice(other_ops)
    if isinstance(sub, BinOp):
        return _replace_at(tree, idx, BinOp(new_op, sub.a, sub.b))
    return _replace_at(tree, idx, UnOp(new_op, sub.a))


# ---------------- Top-level dispatcher ----------------

OP_WEIGHTS = {
    "subtree_swap":       0.25,
    "subtree_crossover":  0.25,
    "constant_jitter":    0.20,
    "term_insert":        0.10,
    "term_delete":        0.10,
    "op_swap":            0.10,
}


def mutate(
    parents: list[Node],
    rng: random.Random,
    feature_names: list[str],
    *,
    max_attempts: int = 5,
) -> Optional[Node]:
    """Produce one valid offspring from a list of parents.

    `parents`: at least 1 (most ops use parents[0]); subtree_crossover needs 2.
    Returns the new tree, or None if all attempts violated constraints.
    """
    if not parents:
        raise ValueError("mutate needs at least one parent")
    feat_set = set(feature_names)
    keys = list(OP_WEIGHTS.keys())
    weights = list(OP_WEIGHTS.values())

    for _ in range(max_attempts):
        op_name = rng.choices(keys, weights=weights, k=1)[0]
        try:
            if op_name == "subtree_swap":
                child = subtree_swap(parents[0], rng, feature_names)
            elif op_name == "subtree_crossover":
                if len(parents) < 2:
                    continue
                child = subtree_crossover(parents[0], parents[1], rng)
            elif op_name == "constant_jitter":
                child = constant_jitter(parents[0], rng)
            elif op_name == "term_insert":
                child = term_insert(parents[0], rng, feature_names)
            elif op_name == "term_delete":
                child = term_delete(parents[0], rng)
            elif op_name == "op_swap":
                child = op_swap(parents[0], rng)
            else:
                continue
        except Exception:
            continue
        if validate_tree(child, feat_set) is None:
            return child
    return None


__all__ = [
    "MAX_DEPTH", "MAX_COMPLEXITY", "MAX_CONST_MAGNITUDE",
    "validate_tree",
    "random_subtree",
    "subtree_swap", "subtree_crossover",
    "constant_jitter", "term_insert", "term_delete", "op_swap",
    "OP_WEIGHTS", "mutate",
]
