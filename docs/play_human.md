# Play the engine through a chess board

A small browser UI that lets a human play against `math-engine-cpp` over UCI.
The eval the engine uses comes from a **strategy spec** — a versioned JSON
file in `strategies/`. Self-play iterations save new specs into the same
directory; the play UI loads whichever you point at.

## Quick start

```
# from repo root, ASCII path
cd C:\dev\symbolic-chess
$env:PYTHONPATH = "src"
python -m symbolic_chess.play
```

Opens an HTTP server at `http://127.0.0.1:8080/`. Default spec is
`strategies/cx13_iter0.json`. Drag pieces; engine answers via the UCI
subprocess at the depth shown on screen.

Options:

```
python -m symbolic_chess.play --strategy strategies/cx13_iter0.json \
                              --port 8080 --depth 5
```

`--engine` overrides the UCI binary path if you want to test a custom build.

## Strategy spec schema

```json
{
  "id": "cx13_iter0",
  "version": 1,
  "provenance": { "method": "...", "iter": 0, "source_doc": "...", "pareto_complexity": 13 },
  "engine": { "uci_binary": "math-engine-cpp/build/math_engine.exe", "default_depth": 5 },
  "features": ["material_net", "W_mobility", "central_BP", "central_BB", "B_king_zone_own_pawns"],
  "material_unit_weights": { "P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0 },
  "constants": { "c_mat_div": 0.0146, "c_denom_offset": 1.3168 },
  "expression": "(material_net / c_mat_div) - ((W_mobility * central_BP) / ((c_denom_offset - B_king_zone_own_pawns) - central_BB))",
  "safe_div": true,
  "output_units": "centipawns_white_perspective"
}
```

Load/save via `symbolic_chess.strategy.store`:

```python
from symbolic_chess.strategy.store import load_strategy, save_strategy
spec = load_strategy("strategies/cx13_iter0.json")
```

`expression` is the human-readable PySR sympy form. The current C++ engine
implements `evaluate_sr_cx13` directly in `math-engine-cpp/src/eval.cpp` —
the JSON spec is the canonical record of *which* expression that function
should match, not a runtime-loaded definition. When self-play iter-1+ produces
a new Pareto winner, save a new `strategies/<id>.json` next to this one and
update the C++ eval to match.

## Architecture

```
  browser (chessboard.js + chess.js, CDN)
       |   HTTP POST /api/move {human_move, depth}
       v
  src/symbolic_chess/play/server.py  (stdlib http.server)
       |   subprocess stdin/stdout (UCI)
       v
  src/symbolic_chess/play/uci_bridge.py
       |
       v
  math-engine-cpp/build/math_engine.exe   (the strategy)
```

Single in-memory game; one engine subprocess held across all moves.
`/api/new` resets the board, `/api/move` accepts a UCI move and returns the
engine's reply. python-chess validates legality on both sides.

## Tests

```
pytest tests/play/
```

- `test_strategy_store.py` — roundtrip + bundled cx13_iter0.json parses.
- `test_uci_bridge.py` — engine returns a legal move from startpos and after
  `1.e4 e5`. Skipped if `math_engine.exe` isn't built.
