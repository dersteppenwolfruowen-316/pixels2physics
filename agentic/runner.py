"""Run a model-written Taichi program and return its frames (RGB, 256 px) up to a frame limit, or why it failed.

Unlike refine/sim.py this does not force determinism (model programs choose their own arch; on the GPU VM ti.gpu is
CUDA). Programs that render through taichi.tools.video.VideoManager stop as soon as enough frames are written;
programs that write a video file themselves (imageio, ...) run to the end.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIZE = 256  # panel resolution; metrics use 128 px (2x2 mean), as the evaluation

PRELUDE = """import os as _os
try:
    from taichi.tools import video as _video
    _max_frames = int(_os.environ.get("P2P_MAX_FRAMES", "0"))
    if _max_frames:
        _write_frame = _video.VideoManager.write_frame
        def _write_frame_and_stop(self, img):
            _write_frame(self, img)
            if self.frame_counter >= _max_frames:
                _os._exit(0)
        _video.VideoManager.write_frame = _write_frame_and_stop
except Exception:
    pass
"""
FFMPEG_SHIM = '#!/bin/sh\nfor a in "$@"; do last=$a; done\n[ -n "$last" ] && : > "$last"\nexit 0\n'
PRELUDE_LINES = PRELUDE.count("\n")


@dataclass
class Rollout:
    frames: np.ndarray | None  # uint8 [n, SIZE, SIZE, 3]
    error: str = ""  # short reason (one line) when frames is None or too short
    detail: str = ""  # traceback tail shown to the model


def static_errors(code: str) -> str:
    """Undefined names found without running the program (pyflakes)."""
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io

        out = io.StringIO()
        check(code, "program.py", Reporter(out, out))
        undefined = [line.split(":", 2)[-1].strip() for line in out.getvalue().splitlines() if "undefined name" in line]
        return "; ".join(dict.fromkeys(undefined))
    except ImportError:
        return ""


def _traceback_tail(stderr: str, lines: int = 12) -> str:
    keep = [l for l in stderr.splitlines() if l.strip() and not re.match(r"\[(Taichi|[A-Z]) ", l)
            and "site-packages" not in l]
    tail = "\n".join(keep[-lines:])
    # Line numbers refer to the program as written, not to the prelude we prepend.
    return re.sub(r'File ".*?program\.py", line (\d+)', lambda m: f'line {max(1, int(m.group(1)) - PRELUDE_LINES)}', tail)


def _decode(inputs: list[str], pattern: bool) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error"] + (["-pattern_type", "glob"] if pattern else []) + ["-i", *inputs,
           "-vf", f"scale={SIZE}:{SIZE}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, SIZE, SIZE, 3)


def decode_video(path: Path) -> np.ndarray:
    return _decode([str(path)], pattern=False)


def run(code: str, max_frames: int, timeout: int = 300) -> Rollout:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return Rollout(None, f"SyntaxError: {e.msg}", f"SyntaxError: {e.msg} (line {e.lineno})")
    with tempfile.TemporaryDirectory(prefix="p2p_agent_") as tmp:
        root = Path(tmp)
        (root / "program.py").write_text(PRELUDE + code, encoding="utf-8")
        shim = root / "_bin"
        shim.mkdir()
        (shim / "ffmpeg").write_text(FFMPEG_SHIM)
        (shim / "ffmpeg").chmod(0o755)
        env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}", "P2P_MAX_FRAMES": str(max_frames)}
        try:
            done = subprocess.run([sys.executable, "program.py"], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
            tail = _traceback_tail(done.stderr)
        except subprocess.TimeoutExpired:
            return Rollout(None, "timeout", f"The program did not render {max_frames} frames within {timeout} s.")
        frame_dirs = Counter(p.parent for p in root.rglob("*.png"))
        videos = sorted((p for p in root.rglob("*.mp4") if p.stat().st_size > 0), key=lambda p: p.stat().st_size)
        try:
            if frame_dirs and frame_dirs.most_common(1)[0][1] >= 2:
                frames = _decode([str(frame_dirs.most_common(1)[0][0] / "*.png")], pattern=True)
            elif videos:
                frames = decode_video(videos[-1])
            else:
                first = tail.splitlines()[-1] if tail else f"exit {done.returncode}, no frames written"
                hints = static_errors(code)  # pyflakes has false positives in Taichi code: a hint, never a gate
                return Rollout(None, first[:150], (tail or first) + (f"\nStatic check: {hints}" if hints else ""))
        except subprocess.CalledProcessError:
            return Rollout(None, "unreadable frames", "The program's frames could not be decoded.")
    if len(frames) < max_frames:
        why = tail.splitlines()[-1] if tail else "the program stopped early"
        return Rollout(frames, f"only {len(frames)} frames: {why}"[:150],
                       f"The program rendered only {len(frames)} of the {max_frames} frames needed.\n{tail}".strip())
    return Rollout(frames[:max_frames])
