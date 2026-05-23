from pathlib import Path

from symbolic_chess.strategy.store import (
    StrategySpec, load_strategy, save_strategy,
)


def test_load_bundled_cx13(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    spec = load_strategy(repo / "strategies" / "cx13_iter0.json")
    assert spec.id == "cx13_iter0"
    assert "material_net" in spec.features
    assert "W_mobility" in spec.features
    assert "c_mat_div" in spec.constants
    assert spec.material_unit_weights["Q"] == 9.0
    assert "material_net" in spec.expression


def test_roundtrip(tmp_path):
    src = Path(__file__).resolve().parents[2] / "strategies" / "cx13_iter0.json"
    spec = load_strategy(src)
    out = tmp_path / "roundtrip.json"
    save_strategy(spec, out)
    spec2 = load_strategy(out)
    assert spec2.to_dict() == spec.to_dict()
