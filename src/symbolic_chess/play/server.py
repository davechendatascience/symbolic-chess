"""Local web board for human vs. math-engine-cpp.

stdlib http.server only (no Flask dep). Single-game in-memory state.
chessboard.js + chess.js load from CDN; we just serve the HTML shell and a
small JSON API that drives a UCI subprocess.

Run:
    python -m symbolic_chess.play [--strategy strategies/cx13_iter0.json]
                                  [--port 8080] [--depth 5]
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import chess

from symbolic_chess.strategy.store import load_strategy
from symbolic_chess.play.uci_bridge import UciEngine


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "board.html"


class GameState:
    """One in-memory game shared by all clients on this server."""

    def __init__(self, engine: UciEngine):
        self.lock = threading.Lock()
        self.engine = engine
        self.board = chess.Board()
        self.moves: list[str] = []
        self.human_color = chess.WHITE  # 'w' default

    def reset(self, human_color: chess.Color) -> None:
        with self.lock:
            self.board = chess.Board()
            self.moves = []
            self.human_color = human_color
            self.engine.new_game()

    def play_human_then_engine(
        self, human_uci: str | None, depth: int
    ) -> tuple[str | None, str, bool]:
        """Apply human move (if any), then ask engine for its move. Returns
        (engine_move_uci_or_None, fen_after, game_over)."""
        with self.lock:
            if human_uci is not None:
                mv = chess.Move.from_uci(human_uci)
                if mv not in self.board.legal_moves:
                    raise ValueError(f"illegal human move {human_uci}")
                self.board.push(mv)
                self.moves.append(human_uci)
            if self.board.is_game_over():
                return None, self.board.fen(), True
            if self.board.turn == self.human_color:
                # human's turn (e.g. new game where human is white and asked for engine move)
                return None, self.board.fen(), False
            engine_uci = self.engine.bestmove("startpos", moves=self.moves, depth=depth)
            mv = chess.Move.from_uci(engine_uci)
            if mv not in self.board.legal_moves:
                raise RuntimeError(f"engine returned illegal move {engine_uci}")
            self.board.push(mv)
            self.moves.append(engine_uci)
            return engine_uci, self.board.fen(), self.board.is_game_over()


def make_handler(state: GameState, strategy_id: str, default_depth: int):
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    page = (
        template.replace("__STRATEGY_ID__", strategy_id)
        .replace("__DEPTH__", str(default_depth))
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter logs
            return

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"})
                return

            if self.path == "/api/new":
                side = (payload.get("human") or "w").lower()
                human_color = chess.WHITE if side.startswith("w") else chess.BLACK
                state.reset(human_color)
                self._send_json(200, {"fen": state.board.fen(), "human": side})
                return

            if self.path == "/api/move":
                human_uci = payload.get("human_move")
                depth = int(payload.get("depth") or default_depth)
                try:
                    engine_uci, fen, over = state.play_human_then_engine(human_uci, depth)
                except Exception as e:
                    self._send_json(400, {"error": str(e)})
                    return
                self._send_json(200, {
                    "engine_move": engine_uci,
                    "fen": fen,
                    "game_over": over,
                })
                return

            self._send_json(404, {"error": "no such endpoint"})

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description="Web board for human vs. math-engine-cpp.")
    p.add_argument(
        "--strategy",
        type=str,
        default=str(REPO_ROOT / "strategies" / "cx13_iter0.json"),
        help="path to strategy JSON spec",
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help="UCI search depth (default: from strategy spec)",
    )
    p.add_argument(
        "--engine",
        type=str,
        default=None,
        help="override path to UCI binary (default: from strategy spec)",
    )
    args = p.parse_args()

    spec = load_strategy(args.strategy)
    binary = args.engine or str(REPO_ROOT / spec.engine.get("uci_binary", ""))
    depth = args.depth or int(spec.engine.get("default_depth", 5))

    print(f"strategy: {spec.id}  expr: {spec.expression}")
    print(f"engine:   {binary}  depth={depth}")

    with UciEngine(binary, default_depth=depth, strategy_path=args.strategy) as engine:
        engine.new_game()
        state = GameState(engine)
        handler = make_handler(state, spec.id, depth)
        httpd = HTTPServer(("127.0.0.1", args.port), handler)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"serving at {url}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
        finally:
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
