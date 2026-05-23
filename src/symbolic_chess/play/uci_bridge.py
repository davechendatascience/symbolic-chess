"""Tiny UCI client for driving math-engine-cpp from Python.

Just enough of the UCI protocol to play a game:
  uci/uciok handshake, isready/readyok, position [fen|startpos] moves..., go depth N.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class UciEngine:
    def __init__(
        self,
        binary: str | Path,
        default_depth: int = 5,
        strategy_path: str | Path | None = None,
    ):
        self.binary = str(Path(binary).resolve())
        if not Path(self.binary).is_file():
            raise FileNotFoundError(f"UCI binary not found: {self.binary}")
        self.default_depth = int(default_depth)
        self.strategy_path = str(Path(strategy_path).resolve()) if strategy_path else None
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "UciEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [self.binary]
        if self.strategy_path:
            cmd += ["--strategy", self.strategy_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("quit")
            self._proc.wait(timeout=2.0)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(
        self,
        fen: str = "startpos",
        moves: Optional[list[str]] = None,
        depth: Optional[int] = None,
    ) -> str:
        if fen == "startpos":
            pos_cmd = "position startpos"
        else:
            pos_cmd = f"position fen {fen}"
        if moves:
            pos_cmd += " moves " + " ".join(moves)
        self._send(pos_cmd)
        d = self.default_depth if depth is None else int(depth)
        self._send(f"go depth {d}")
        line = self._read_until_prefix("bestmove")
        # "bestmove e2e4 [ponder ...]"
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f"bad bestmove line: {line!r}")
        return parts[1]

    # --- internal ----
    def _send(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_until(self, exact: str, timeout_lines: int = 10_000) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        for _ in range(timeout_lines):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine closed before {exact!r}")
            if line.strip() == exact:
                return line.strip()
        raise RuntimeError(f"never saw {exact!r}")

    def _read_until_prefix(self, prefix: str, timeout_lines: int = 100_000) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        for _ in range(timeout_lines):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine closed before line starting {prefix!r}")
            if line.startswith(prefix):
                return line.strip()
        raise RuntimeError(f"never saw line starting {prefix!r}")
