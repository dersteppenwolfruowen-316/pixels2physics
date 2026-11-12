"""Verifier: compare a rollout with the observed frames, per stage, and write the feedback the model sees.

Only the 16 observed frames (all before the held-out start) are ever used here. Scores use the evaluation's
32x32 occupancy IoU on 128 px frames; positions and sizes come from the foreground mask in normalized scene
coordinates (x to the right, y up, both in [0, 1]).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import occupancy  # noqa: E402


@dataclass(frozen=True)
class Stage:
    name: str
    positions: tuple[int, ...]  # observed-frame positions (0..15) that judge this stage
    panels: tuple[int, ...]  # positions shown as comparison images
    focus: str


STAGES = (
    Stage("geometry", (0, 1), (0, 1),
          "Fix the GEOMETRY first: the initial positions, sizes and shapes of the objects (frames 1-2). "
          "Keep materials, velocities and gravity unless they are clearly the cause."),
    Stage("motion", (0, 1, 2, 3, 4), (0, 2, 4),
          "Now fix the MOTION: the initial velocities of the objects, judged on frames 1-5. "
          "Keep the geometry you already have."),
    Stage("force", tuple(range(16)), (0, 5, 10, 15),
          "Now fix the FORCES over the whole observed window (frames 1-16): gravity and any other external forces "
          "or boundary effects. Keep geometry and initial velocities."),
)


def gray128(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    n, h, w = g.shape
    return g.reshape(n, 128, h // 128, 128, w // 128).mean(axis=(2, 4))


def masks(rgb: np.ndarray) -> np.ndarray:
    g = gray128(rgb)
    border = np.concatenate([g[0, 0], g[0, -1], g[0, :, 0], g[0, :, -1]])
    return np.abs(g - np.median(border)) > 30


def ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    oa, ob = occupancy(a), occupancy(b)
    union = (oa | ob).reshape(len(oa), -1).sum(axis=1)
    return np.where(union > 0, (oa & ob).reshape(len(oa), -1).sum(axis=1) / np.maximum(union, 1), 1.0)


def shape_stats(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """(centroid x, centroid y, width, height) of a foreground mask, normalized, y up."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    n = mask.shape[0]
    return xs.mean() / n, 1 - ys.mean() / n, (xs.max() - xs.min() + 1) / n, (ys.max() - ys.min() + 1) / n


@dataclass
class Comparison:
    """A rollout compared with the observation on all 16 observed frames."""
    iou: np.ndarray  # [16]
    obs: list
    sim: list
    obs_change: float | None = None  # fraction of pixels whose foreground state differs between frames 1 and 16
    sim_change: float | None = None

    def loss(self, stage: Stage) -> float:
        return float(1 - self.iou[list(stage.positions)].mean())


def compare(observed: np.ndarray, simulated: np.ndarray, indices: list[int]) -> Comparison:
    """observed: [16, S, S, 3] observed frames; simulated: full rollout frames; indices: video frame of each position."""
    sim = simulated[indices]
    mo, ms = masks(observed), masks(sim)
    return Comparison(ious(mo, ms), [shape_stats(m) for m in mo], [shape_stats(m) for m in ms],
                      float((mo[0] ^ mo[-1]).mean()), float((ms[0] ^ ms[-1]).mean()))


def composite(observed: np.ndarray, simulated: np.ndarray, label: str) -> Image.Image:
    """observed | simulated | overlay (observed red, simulated green, both yellow), with a caption."""
    mo, ms = masks(observed[None])[0], masks(simulated[None])[0]
    overlay = np.full((128, 128, 3), 255, np.uint8)
    overlay[mo & ~ms] = (220, 40, 40)
    overlay[ms & ~mo] = (40, 170, 40)
    overlay[mo & ms] = (230, 190, 0)
    size = observed.shape[0]
    overlay = np.asarray(Image.fromarray(overlay).resize((size, size), Image.NEAREST))
    caption = 18
    canvas = Image.new("RGB", (3 * size + 8, size + caption), (255, 255, 255))
    for i, panel in enumerate((observed, simulated, overlay)):
        canvas.paste(Image.fromarray(panel), (i * (size + 4), caption))
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate((f"observed {label}", f"your simulation {label}", "overlay: red=observed only, green=simulation only")):
        draw.text((i * (size + 4) + 3, 3), text, fill=(0, 0, 0))
    return canvas


def _fmt(s) -> str:
    return "nothing visible" if s is None else f"center ({s[0]:.2f}, {s[1]:.2f}), size {s[2]:.2f} x {s[3]:.2f}"


def stage_text(c: Comparison, stage: Stage, times: list[str]) -> str:
    shown = sorted(set(stage.panels) | set(stage.positions[:: max(1, len(stage.positions) // 6)]))
    rows = [f"- frame {p + 1} ({times[p]}): IoU {c.iou[p]:.2f}; observed {_fmt(c.obs[p])}; simulated {_fmt(c.sim[p])}"
            for p in shown]
    if stage.name != "geometry" and c.obs_change and c.sim_change is not None and c.sim_change < 0.2 * c.obs_change:
        rows.append("- Between frame 1 and frame 16 the observation changes a lot but your simulation barely changes: "
                    "something that should move (or appear) does not.")
    deltas = [(s[0] - o[0], s[1] - o[1], s[2] - o[2], s[3] - o[3]) for p in stage.positions
              if (o := c.obs[p]) is not None and (s := c.sim[p]) is not None]
    summary = ""
    if deltas:
        dx, dy, dw, dh = np.mean(deltas, axis=0)
        words = []
        if abs(dx) >= 0.02:
            words.append(f"{abs(dx):.2f} too far {'right' if dx > 0 else 'left'}")
        if abs(dy) >= 0.02:
            words.append(f"{abs(dy):.2f} too {'high' if dy > 0 else 'low'}")
        if abs(dw) >= 0.02:
            words.append(f"{abs(dw):.2f} too {'wide' if dw > 0 else 'narrow'}")
        if abs(dh) >= 0.02:
            words.append(f"{abs(dh):.2f} too {'tall' if dh > 0 else 'short'}")
        summary = ("On average the simulated foreground is " + ", ".join(words) + "." if words
                   else "On average the simulated foreground's position and extent match the observation.")
    return (f"Mean IoU on these frames: {1 - c.loss(stage):.2f} (1 = perfect). Coordinates are normalized to the frame "
            "(x to the right, y up, 0 to 1); they include static walls and floors.\n" + "\n".join(rows) + "\n" + summary)
