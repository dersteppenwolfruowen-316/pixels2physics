"""Generate predictions and score them on two tracks.

    # 1. predictions (vLLM; a LoRA adapter is merged first)
    python scripts/evaluate.py generate --dataset data/sft_v1/swift/c4_no_state/test.jsonl \
        --adapter runs/c4_no_state__weighted/vX/checkpoint-best --out runs/eval/c4_no_state/pred.jsonl
    python scripts/evaluate.py generate ... --num-samples 4 --temperature 0.7 --out runs/eval/c4_no_state__bo4/pred.jsonl
    # reference predictions: ground truth (simulation ceiling) and train-set mode (blind baseline)
    python scripts/evaluate.py oracle   --dataset data/sft_v1/swift/c4_no_state/test.jsonl --out runs/eval/oracle/pred.jsonl
    python scripts/evaluate.py baseline --data data/sft_v1 --out runs/eval/mode/pred.jsonl
    # 2. scores (--simulate executes programs: untrusted code, use a VM or container)
    python scripts/evaluate.py score --pred runs/eval/c4_no_state/pred.jsonl --simulate --fill
    python scripts/evaluate.py score --pred runs/eval/c4_no_state__bo4/pred.jsonl --simulate --best-of-n
    # 3. tables across runs
    python scripts/evaluate.py compare runs/eval/*/summary.json

Physics track: can the model infer the hidden quantities from video?
    parameters      geometry MAE (domain units), motion relative error, gravity log10 error, against the truth
    filled rollout  (--fill) the predicted hidden values replace the true ones in the ground-truth program, which
                    then runs: the program is always right, so the rollout measures only the inferred physics.
                    Restricted to scenes where every hidden value appears as a literal in the program.
Code track: can the model write the whole program?
    rollout         run the model's own program (--best-of-n: the first of N samples that runs)
Rollouts are compared with the ground-truth video, mainly on the held-out last 30% (never shown to the
model): occupancy IoU, windowed motion-activity error (W-MAE), object collapse (OCS); programs that fail
get the worst value of each. Every number carries a 95% bootstrap CI resampled over families (parent_id).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PHYSICS = re.compile(r"<physics>(.*?)</physics>", re.S)
CODE = re.compile(r"```python\n(.*?)(?:```|$)", re.S)
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")
GROUPS = ("material", "geometry", "motion")
REL_TOL = 1e-3
# A numeric field counts as recovered within tolerance: geometry 0.02 of the unit domain (relative for
# fields > 1), motion 10% relative.
TOLERANCE = {"material": lambda t: REL_TOL * abs(t), "geometry": lambda t: 0.02 * max(1.0, abs(t)),
             "motion": lambda t: 0.1 * max(0.1, abs(t))}
HEADLINE = (
    "physics_valid", "code_parses", "json_code_consistency",
    "cm_acc", "material_exact", "density_log10_err", "youngs_log10_err", "poisson_abs_err", "material_key_recall",
    "geometry_acc_tol", "geometry_mae_unit", "geometry_key_recall", "motion_acc_tol", "motion_rel_err", "motion_key_recall",
    "gravity_log_err", "fill_fallback", "fill_sim_success", "fill_iou_heldout", "fill_wmae_heldout", "fill_ocs",
    "sim_success", "iou_observed", "iou_heldout", "wmae_heldout", "ocs", "iou_heldout_ok", "wmae_heldout_ok",
)
# The two main tables (`compare --all` prints every metric). The inferred quantities are the kinematic state
# (positions, velocities) and the gravity; materials are given as conditions.
PHYSICS_CORE = ("geometry_mae_unit", "motion_rel_err", "gravity_log_err", "geometry_key_recall",
                "fill_iou_heldout", "fill_wmae_heldout", "fill_ocs")
CODE_CORE = ("sim_success", "iou_heldout", "wmae_heldout", "ocs")
# Hidden quantities per config; the filled rollout replaces exactly these in the ground-truth program.
HIDDEN = {"c0_video_only": ("geometry", "motion"), "c1_full_aux": (), "c2_no_position": ("geometry",),
          "c3_no_material": (), "c4_no_state": ("geometry", "motion", "gravity")}
# Every config is filled-scored on the same scenes: those whose C4-hidden values are all literals in the program.
FILL_SCOPE = ("geometry", "motion", "gravity")
TRIVIAL = {0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}  # too common in MPM code to locate a field by its value
# A program that fails to run or render gets the worst rollout scores, so failures count against a model
# instead of shrinking the set its averages are taken over.
FAILED_ROLLOUT = {"iou_observed": 0.0, "iou_heldout": 0.0, "wmae_heldout": 1.0, "ocs": 1.0}
ACTIVITY_GRID, ACTIVITY_WINDOW = 32, 15  # 4x4-pixel blocks of the 128px frame; 0.5 s at 30 fps
# IoU and OCS use the same 4x4-pixel blocks: a block is occupied if any pixel in it is foreground. Particles render
# as sparse dots placed at random, so pixel masks of the ground-truth program itself overlap only ~0.7.
MASK_GRID = 32


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ----------------------------------------------------------------------------- predictions

def generate(args: argparse.Namespace) -> None:
    from swift.infer_engine import InferRequest, RequestConfig

    rows = read_jsonl(args.dataset)
    model = args.model
    if args.adapter and args.backend == "vllm":
        merged = Path(f"{args.adapter.rstrip('/')}-merged")
        if not merged.is_dir():
            subprocess.run(["swift", "export", "--adapters", args.adapter, "--merge_lora", "true", "--use_hf", "true"], check=True)
        model = str(merged)
    if args.backend == "vllm":
        from swift.infer_engine import VllmEngine

        engine = VllmEngine(model, use_hf=True, max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 64},
                            gpu_memory_utilization=0.85)
    else:
        from swift.infer_engine import TransformersEngine

        engine = TransformersEngine(model, adapters=[args.adapter] if args.adapter else None, use_hf=True, max_batch_size=4)
    requests = [InferRequest(messages=[m for m in row["messages"] if m["role"] != "assistant"], images=row["images"]) for row in rows]
    # Greedy by default; --num-samples N draws N independent samples per scene for best-of-N.
    temperature = args.temperature if args.num_samples > 1 else 0.0
    # --physics-only stops after the <physics> block (controls score parameters only): ~7x fewer tokens.
    # No seed when sampling: with a fixed seed some vLLM versions give all n samples of a request the same draw.
    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=temperature, top_p=args.top_p if temperature else None,
                           n=args.num_samples, seed=None if temperature else 0, stop=["</physics>"] if args.physics_only else [])
    responses = engine.infer(requests, config, use_tqdm=True)
    out = []
    for row, r in zip(rows, responses):
        texts = [c.message.content + ("</physics>" if args.physics_only else "") for c in r.choices]
        out.append({"scene_id": row["scene_id"], "response": texts[0], **({"responses": texts} if len(texts) > 1 else {})})
    write_jsonl(args.out, out)


def oracle(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.dataset)
    write_jsonl(args.out, [{"scene_id": row["scene_id"], "response": row["messages"][-1]["content"]} for row in rows])


def mode_guesses(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Blind guess per test scene: the constitutive model as the train-set mode for the object's role, then every
    other field as the train-set mode among objects with that role and that guessed model, falling back to the
    same path, then the same key name and shape (paths are often unique to a family); gravity: the mode."""
    pools: dict[tuple, Counter] = defaultdict(Counter)
    for meta in train:
        for role, obj in zip(meta["roles"], meta["physics"]):
            cm = constitutive_model(obj)
            for group in GROUPS:
                for path, value in obj.get(group, {}).items():
                    shape = len(value) if isinstance(value, list) else 0
                    for key in ((role, cm, group, path), (role, group, path), (group, path), (group, path.rsplit(".", 1)[-1], shape)):
                        pools[key][json.dumps(value)] += 1
    gravity_mode = json.loads(Counter(json.dumps(meta.get("gravity", {})) for meta in train).most_common(1)[0][0])
    guesses = {}
    for meta in test:
        predicted = []
        for role, obj in zip(meta["roles"], meta["physics"]):
            cm_path = next((p for p in obj.get("material", {}) if p.endswith("constitutive_model")), None)
            cm_pool = pools[(role, "material", cm_path)] if cm_path else None
            cm = json.loads(cm_pool.most_common(1)[0][0]) if cm_pool else None
            guess = {"id": obj["id"]}
            for group in GROUPS:
                fields = {}
                for path, value in obj.get(group, {}).items():
                    shape = len(value) if isinstance(value, list) else 0
                    keys = ((role, cm, group, path), (role, group, path), (group, path), (group, path.rsplit(".", 1)[-1], shape))
                    if pool := next((pools[k] for k in keys if pools[k]), None):
                        fields[path] = json.loads(pool.most_common(1)[0][0])
                guess[group] = fields
            predicted.append(guess)
        predicted.append({"id": "scene", "environment": gravity_mode})
        guesses[meta["scene_id"]] = predicted
    return guesses


def baseline(args: argparse.Namespace) -> None:
    train, test = read_jsonl(args.data / "meta" / "train.jsonl"), read_jsonl(args.data / "meta" / f"{args.split}.jsonl")
    write_jsonl(args.out, [{"scene_id": scene, "response": f"<physics>\n{json.dumps(guess)}\n</physics>"}
                           for scene, guess in mode_guesses(train, test).items()])


# ----------------------------------------------------------------------------- L0 / L1

def parse_response(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    physics = None
    if match := PHYSICS.search(text):
        try:
            value = json.loads(match.group(1))
            physics = value if isinstance(value, list) else None
        except json.JSONDecodeError:
            pass
    code = match.group(1) if (match := CODE.search(text)) else None
    return physics, code


def constitutive_model(obj: dict[str, Any]) -> str | None:
    return next((v for p, v in obj.get("material", {}).items() if p.endswith("constitutive_model")), None)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numbers(value: Any) -> list[float] | None:
    if is_number(value):
        return [float(value)]
    if isinstance(value, list) and value and all(is_number(v) for v in value):
        return [float(v) for v in value]
    return None


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-9)


def compare_fields(truth: dict[str, Any], predicted: dict[str, Any], group: str) -> list[dict[str, Any]]:
    """One item per ground-truth field path: present, exact, within tolerance, abs/log errors where numeric."""
    items = []
    for path, value in truth.items():
        item = {"path": path, "key": path.rsplit(".", 1)[-1].lower(), "present": path in predicted}
        guess = predicted.get(path)
        t, p = numbers(value), numbers(guess)
        if t is not None and p is not None and len(t) == len(p):
            item["exact"] = all(close(a, b) for a, b in zip(t, p))
            item["tol_ok"] = all(abs(a - b) <= TOLERANCE[group](a) for a, b in zip(t, p))
            # Relative vector error, unit-free (fields mix m/s, rad/s, deg/frame); floor 0.1 for zero vectors,
            # capped at 1 so a single wild value cannot dominate the mean.
            norm = lambda v: math.sqrt(sum(x * x for x in v))
            item["rel_err"] = min(1.0, norm([b - a for a, b in zip(t, p)]) / max(norm(t), 0.1))
            item["abs_err"] = mean([abs(a - b) for a, b in zip(t, p)])
            item["unit_abs_err"] = mean([abs(a - b) for a, b in zip(t, p) if abs(a) <= 1.5])  # domain coordinates
            if len(t) == 1 and t[0] > 0 and p[0] > 0:
                item["log10_err"] = abs(math.log10(p[0] / t[0]))
        else:
            item["exact"] = item["tol_ok"] = guess == value
        items.append(item)
    return items


def gravity_strength(environment: Any) -> float | None:
    """|gravity vector| times any numeric gravity multiplier (gravity_multiplier, gravity_scaling_on_grid, ...)."""
    if not isinstance(environment, dict) or (vector := numbers(environment.get("gravity"))) is None:
        return None
    strength = math.sqrt(sum(x * x for x in vector))
    for key, value in environment.items():
        if key != "gravity" and is_number(value):
            strength *= abs(value)
    return strength


def gravity_log_err(truth: dict[str, Any], predicted: list[dict[str, Any]] | None) -> float | None:
    """|log10(g_pred / g_true)| of the gravity strength, capped at 2 (two orders of magnitude, beyond the spread
    of the data: |g| from 1 to 147); an unusable prediction counts as 2."""
    scene = next((o for o in predicted or [] if isinstance(o, dict) and o.get("id") == "scene"), {})
    true, guess = gravity_strength(truth), gravity_strength(scene.get("environment"))
    if not true:
        return None
    return min(2.0, abs(math.log10(guess / true))) if guess else 2.0


def complete_with(predicted: list[dict[str, Any]] | None, fallback: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """The prediction with every missing or misshaped true field (and gravity) taken from the blind guess, so a
    model cannot lower its errors by leaving the fields it is unsure of out of the JSON."""
    by_id = {o["id"]: {k: (dict(v) if isinstance(v, dict) else v) for k, v in o.items()}
             for o in predicted or [] if isinstance(o, dict) and isinstance(o.get("id"), str)}
    guesses = {o["id"]: o for o in fallback}
    fields = [(o["id"], g, p, v) for o in meta["physics"] for g in GROUPS for p, v in o.get(g, {}).items()]
    fields += [("scene", "environment", p, v) for p, v in meta.get("gravity", {}).items()]
    for object_id, group, path, value in fields:
        guess, truth = predicted_value(list(by_id.values()), object_id, group, path), numbers(value)
        if truth is not None and (guess is None or len(guess) != len(truth)):
            if (replacement := guesses.get(object_id, {}).get(group, {}).get(path)) is not None:
                obj = by_id.setdefault(object_id, {"id": object_id})
                obj[group] = {**(obj[group] if isinstance(obj.get(group), dict) else {}), path: replacement}
    return list(by_id.values())


def score_physics(truth: list[dict[str, Any]], predicted: list[dict[str, Any]] | None) -> dict[str, Any]:
    by_id = {o.get("id"): o for o in predicted or [] if isinstance(o, dict)}
    items: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}
    for obj in truth:
        guess = by_id.get(obj["id"], {})
        for group in GROUPS:
            fields = guess.get(group) if isinstance(guess.get(group), dict) else {}
            items[group] += compare_fields(obj.get(group, {}), fields, group)

    def pick(group: str, pattern: str, field: str) -> float | None:
        return mean([i.get(field) for i in items[group] if i["present"] and re.search(pattern, i["key"])])

    m = items["material"]
    return {
        "cm_acc": mean([float(i["exact"]) for i in m if i["key"] == "constitutive_model"]),
        "material_exact": mean([float(i["exact"]) for i in m]),
        "density_log10_err": pick("material", r"density|^rho$", "log10_err"),
        "youngs_log10_err": pick("material", r"young|^e$", "log10_err"),
        "poisson_abs_err": pick("material", r"poisson|^nu$", "abs_err"),
        "material_key_recall": mean([float(i["present"]) for i in m]),
        "geometry_mae_unit": mean([i.get("unit_abs_err") for i in items["geometry"] if i["present"]]),
        "motion_rel_err": mean([i.get("rel_err") for i in items["motion"] if i["present"]]),
        **{f"{g}_acc_tol": mean([float(i["tol_ok"]) for i in items[g]]) for g in ("geometry", "motion")},
        **{f"{g}_key_recall": mean([float(i["present"]) for i in items[g]]) for g in ("geometry", "motion")},
    }


def json_code_consistency(physics: list[dict[str, Any]] | None, code: str | None) -> float | None:
    if not physics or not code:
        return None
    wanted = {float(f"{float(n):.4g}") for n in NUMBER.findall(json.dumps(physics))} - {0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}
    found = {float(f"{float(n):.4g}") for n in NUMBER.findall(code)}
    return len(wanted & found) / len(wanted) if wanted else None


# ----------------------------------------------------------------------------- physics track: filled program

def hidden_fields(meta: dict[str, Any], groups: tuple[str, ...]) -> list[tuple[str, str, str, Any]]:
    """(object id, group, path, true value) of every hidden field; gravity lives on a "scene" entry."""
    fields = [(o["id"], g, p, v) for o in meta["physics"] for g in groups if g != "gravity" for p, v in o.get(g, {}).items()]
    if "gravity" in groups:
        fields += [("scene", "environment", p, v) for p, v in meta.get("gravity", {}).items()]
    return fields


def literal_key(value: float) -> float:
    return float(f"{abs(value):.4g}")  # signs are separate tokens in code: -9.8 is "-" then "9.8"


def fill_scope_complete(meta: dict[str, Any], code: str) -> bool:
    """Every non-trivial value hidden in C4 appears as a literal in the program, so filling can replace it."""
    literals = {literal_key(float(n)) for n in NUMBER.findall(code)}
    values = [x for *_, v in hidden_fields(meta, FILL_SCOPE) for x in numbers(v) or []]
    return all(literal_key(x) in literals for x in values if literal_key(x) not in TRIVIAL)


def predicted_value(physics: list[dict[str, Any]] | None, object_id: str, group: str, path: str) -> list[float] | None:
    section = next((o.get(group) for o in physics or [] if isinstance(o, dict) and o.get("id") == object_id), None)
    return numbers(section.get(path)) if isinstance(section, dict) else None


def fill_program(meta: dict[str, Any], code: str, predicted: list[dict[str, Any]] | None, groups: tuple[str, ...],
                 fallback: list[dict[str, Any]]) -> tuple[str | None, float]:
    """The ground-truth program with every literal equal to a hidden true value replaced by the predicted value,
    and the fraction of hidden fields that fell back to the blind guess. A field missing from the prediction or of
    the wrong shape takes the train-set mode (`fallback`), never the true value; with no usable guess either, the
    program is None (scored as a failed rollout). Values in TRIVIAL are left alone: they cannot be told apart from
    boilerplate constants."""
    replace: dict[float, float] = {}
    fields = used_fallback = 0
    for object_id, group, path, value in hidden_fields(meta, groups):
        truth = numbers(value)
        if truth is None:
            continue  # non-numeric (e.g. a shape type): not inferred as a number
        fields += 1
        guess = predicted_value(predicted, object_id, group, path)
        if guess is None or len(guess) != len(truth):
            used_fallback += 1
            guess = predicted_value(fallback, object_id, group, path)
            if guess is None or len(guess) != len(truth):
                return None, used_fallback / fields
        for t, g in zip(truth, guess):
            if literal_key(t) not in TRIVIAL:
                replace.setdefault(literal_key(t), abs(g))

    def substitute(match: re.Match) -> str:
        text = match.group()
        if (new := replace.get(literal_key(float(text)))) is None:
            return text
        if not re.search(r"[.eE]", text) and float(new).is_integer():
            return str(int(new))  # keep integer literals integer (array sizes, ranges)
        formatted = f"{new:.6g}"
        return formatted if re.search(r"[.eE]", formatted) else formatted + ".0"

    return NUMBER.sub(substitute, code), used_fallback / fields if fields else 0.0


# ----------------------------------------------------------------------------- L2

def decode_gray(video: Path, size: int = 128):
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vf", f"scale={size}:{size}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size).astype(np.int16)


def foreground(frames):
    import numpy as np

    border = np.concatenate([frames[0, 0], frames[0, -1], frames[0, :, 0], frames[0, :, -1]])
    return np.abs(frames - np.median(border)) > 30


def occupancy(masks):
    """MASK_GRID x MASK_GRID blocks, occupied if any pixel in the block is foreground."""
    n, h, w = masks.shape
    return masks.reshape(n, MASK_GRID, h // MASK_GRID, MASK_GRID, w // MASK_GRID).any(axis=(2, 4))


def mask_iou(truth, predicted, indices: range) -> float:
    scores = []
    for i in indices:
        if i >= len(predicted):
            scores.append(0.0)  # the program stopped early: missing frames count as misses
            continue
        union = (truth[i] | predicted[i]).sum()
        scores.append(float((truth[i] & predicted[i]).sum() / union) if union else 1.0)
    return sum(scores) / len(scores)


def motion_activity(frames):
    """Per frame: mean absolute intensity change since the previous frame on a coarse block grid, smoothed over a
    sliding window. Coarse blocks make it insensitive to particle-level jitter and to exact positions."""
    import numpy as np

    n, h, w = frames.shape
    coarse = frames.reshape(n, ACTIVITY_GRID, h // ACTIVITY_GRID, ACTIVITY_GRID, w // ACTIVITY_GRID).mean(axis=(2, 4)) / 255.0
    activity = np.concatenate([[0.0], np.abs(np.diff(coarse, axis=0)).mean(axis=(1, 2))])
    return np.convolve(activity, np.ones(ACTIVITY_WINDOW) / ACTIVITY_WINDOW, mode="same")


def windowed_motion_error(truth, predicted, start: int) -> float:
    """W-MAE in [0, 1]: sum_t |a_pred - a_true| / (sum_t a_pred + sum_t a_true) over the held-out frames."""
    import numpy as np

    if len(predicted) < len(truth):  # a program that stopped early is frozen on its last frame
        predicted = np.concatenate([predicted, np.repeat(predicted[-1:], len(truth) - len(predicted), axis=0)])
    a_true, a_pred = motion_activity(truth)[start:], motion_activity(predicted[: len(truth)])[start:]
    return float(np.abs(a_pred - a_true).sum() / (a_pred.sum() + a_true.sum() + 1e-6))


def area_loss(masks, start: int, length: int) -> float:
    """Largest relative loss of foreground area after `start`; frames past the end of the video count as empty."""
    areas = [float(masks[t].sum()) if t < len(masks) else 0.0 for t in range(start, length)]
    return 1.0 if areas[0] == 0 else max(0.0, max(1 - a / areas[0] for a in areas))


def simulate(meta: dict[str, Any], code: str, timeout: int, video_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="p2p_sim_") as tmp:
        root = Path(tmp)
        (root / "program.py").write_text(code, encoding="utf-8")
        # Taichi's VideoManager saves PNG frames, then encodes them with ffmpeg at -crf 1, which takes minutes
        # per scene. A stand-in ffmpeg on the program's PATH skips that; the frames are encoded quickly below.
        # It creates an empty output file: VideoManager also builds videos mid-run (frames 4, 8, 16, ...) and
        # deletes the intermediate palette.png, so a stand-in that writes nothing crashes the program.
        shim = root / "_bin"
        shim.mkdir()
        (shim / "ffmpeg").write_text('#!/bin/sh\nfor a in "$@"; do last=$a; done\n[ -n "$last" ] && : > "$last"\nexit 0\n')
        (shim / "ffmpeg").chmod(0o755)
        env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}"}
        # Taichi's CPU backend (macOS) occasionally dies from a signal (SIGBUS) on programs that run fine; one retry.
        for _ in range(2):
            try:
                run = subprocess.run([sys.executable, "program.py"], cwd=tmp, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, errors="replace", timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                return {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": "timeout"}
            if run.returncode >= 0:
                break
        # Last stderr line that is not a Taichi log line (e.g. "NameError: ..."), to tell crashes apart.
        lines = [line for line in run.stderr.splitlines() if line.strip() and not re.match(r"\[[A-Z] ", line)]
        crash = (lines[-1][:200] if lines else f"exit {run.returncode}") if run.returncode else ""
        video_dir.mkdir(parents=True, exist_ok=True)
        kept = video_dir / f"{meta['scene_id']}.mp4"
        frame_dirs = Counter(p.parent for p in root.rglob("*.png"))
        videos = sorted((p for p in root.rglob("*.mp4") if p.stat().st_size > 0), key=lambda p: p.stat().st_size)
        if frame_dirs and frame_dirs.most_common(1)[0][1] >= 10:
            frames = frame_dirs.most_common(1)[0][0]
            encode = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", "30", "-pattern_type", "glob", "-i", str(frames / "*.png"),
                                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(kept)],
                                    capture_output=True, timeout=300, check=False)
            if encode.returncode:
                return {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": "encode_failed"}
        elif videos:  # programs that write the video themselves (e.g. imageio)
            shutil.copy(videos[-1], kept)
        else:
            return {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": f"no_video: {crash}" if crash else "no_video"}
    gray_true = decode_gray(Path(meta["video_path"]))
    try:
        gray_pred = decode_gray(kept)
    except subprocess.CalledProcessError:
        gray_pred = []
    if len(gray_pred) == 0:
        return {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": "unreadable_video"}
    truth, predicted = occupancy(foreground(gray_true)), occupancy(foreground(gray_pred))
    held = meta["heldout_start_frame"]
    return {
        "sim_success": 1.0,
        "iou_observed": mask_iou(truth, predicted, range(0, held, 3)),
        "iou_heldout": mask_iou(truth, predicted, range(held, len(truth), 3)),
        "wmae_heldout": windowed_motion_error(gray_true, gray_pred, held),
        # Object collapse: area the prediction loses after the held-out start, beyond what the true video loses.
        "ocs": max(0.0, area_loss(predicted, held, len(truth)) - area_loss(truth, held, len(truth))),
    }


# ----------------------------------------------------------------------------- L3 + aggregation

def material_signature(physics: list[dict[str, Any]] | None) -> str:
    """Values of the canonical material fields per object; field names are ignored (they vary across scenes)."""
    def canonical(material: dict[str, Any]) -> list:
        keys = {p.rsplit(".", 1)[-1].lower(): v for p, v in material.items()}
        return [next((str(v) for k, v in keys.items() if re.search(pattern, k)), None)
                for pattern in (r"constitutive_model", r"density|^rho$", r"young|^e$", r"poisson|^nu$")]

    objects = [o for o in physics or [] if isinstance(o, dict) and isinstance(o.get("material"), dict)]
    return json.dumps(sorted((str(o.get("id")), canonical(o["material"])) for o in objects))


def sensitivity(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """Sibling pairs with identical object roles: material differs in truth -> does the prediction differ?
    Same true material -> does the prediction stay the same?"""
    by_family = defaultdict(list)
    for s in scores:
        by_family[s["parent_id"]].append(s)
    changed, stable = [], []
    for siblings in by_family.values():
        for i, a in enumerate(siblings):
            for b in siblings[i + 1:]:
                if a["roles"] != b["roles"]:
                    continue
                pred_differs = float(a["pred_material"] != b["pred_material"])
                (changed if a["true_material"] != b["true_material"] else stable).append(pred_differs)
    return {
        "material_change_detected": mean(changed),
        "material_stable_under_perturbation": None if not stable else 1 - mean(stable),
        "n_changed_pairs": len(changed),
    }


def bootstrap(scores: list[dict[str, Any]], metric: str, rounds: int, seed: int = 0) -> dict[str, Any]:
    values = [(s["parent_id"], s[metric]) for s in scores if s.get(metric) is not None]
    if not values:
        return {"mean": None, "n": 0}
    families = defaultdict(list)
    for family, value in values:
        families[family].append(value)
    keys, rng, means = list(families), random.Random(seed), []
    for _ in range(rounds):
        sample = [v for k in rng.choices(keys, k=len(keys)) for v in families[k]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean": mean([v for _, v in values]), "ci95": [means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]], "n": len(values)}


def parsed_code(code: str | None) -> str | None:
    try:
        return code if code is not None and ast.parse(code) is not None else None
    except SyntaxError:
        return None


def side_by_side(out: Path, metas: dict[str, dict[str, Any]], scores: list[dict[str, Any]], prefix: str, video_dir: Path) -> None:
    """Ground truth (left) | rollout (right) for the best, median and worst successful rollout by held-out IoU."""
    done = sorted((s for s in scores if s.get(f"{prefix}sim_success")), key=lambda s: -s[f"{prefix}iou_heldout"])
    picks = {"best": done[0], "median": done[len(done) // 2], "worst": done[-1]} if done else {}
    for tag, s in picks.items():
        target = out / "side_by_side" / f"{prefix}{tag}__{s['scene_id']}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", metas[s["scene_id"]]["video_path"], "-i", str(video_dir / f"{s['scene_id']}.mp4"),
                        "-filter_complex", "[0:v]scale=384:384,setsar=1[a];[1:v]scale=384:384,setsar=1[b];[a][b]hstack=inputs=2",
                        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)], check=False, timeout=300)


def score(args: argparse.Namespace) -> None:
    metas = {m["scene_id"]: m for m in read_jsonl(args.data / "meta" / f"{args.split}.jsonl")}
    predictions = read_jsonl(args.pred)
    out = args.out_dir or args.pred.parent
    config = args.config or next((c for c in HIDDEN if c in str(args.pred)), None)
    hidden = HIDDEN.get(config, ())
    fallback = mode_guesses(read_jsonl(args.data / "meta" / "train.jsonl"), list(metas.values()))
    scores = []
    for row in predictions:
        meta = metas[row["scene_id"]]
        physics, code = parse_response(row["response"])
        code = parsed_code(code)
        candidates = row.get("responses", [row["response"]]) if args.best_of_n else [row["response"]]
        # Errors are measured on the prediction completed with blind guesses; key recall on the raw prediction.
        completed = complete_with(physics, fallback[meta["scene_id"]], meta)
        recall = {k: v for k, v in score_physics(meta["physics"], physics).items() if k.endswith("key_recall")}
        scores.append({
            "scene_id": meta["scene_id"], "parent_id": meta["parent_id"], "variant": meta["variant"],
            "physics_valid": float(physics is not None),
            "code_parses": float(code is not None),
            "json_code_consistency": json_code_consistency(physics, code),
            **score_physics(meta["physics"], completed), **recall,
            "gravity_log_err": gravity_log_err(meta.get("gravity", {}), completed) if "gravity" in hidden else None,
            "roles": meta["roles"],
            "true_material": material_signature(meta["physics"]),
            "pred_material": material_signature(physics),
            "_codes": [parsed_code(parse_response(text)[1]) for text in candidates],
            "_physics": physics,
        })
    if args.simulate:
        def run(s: dict[str, Any]) -> dict[str, Any]:
            # Code track: the model's own program; with --best-of-n the first sample that runs (selection never
            # looks at the ground-truth video).
            result = {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": "no_code"}
            for index, code in enumerate(s["_codes"]):
                if code is None:
                    continue
                result = {**simulate(metas[s["scene_id"]], code, args.sim_timeout, out / "videos"), "sample": index}
                if result["sim_success"]:
                    break
            if result["sim_success"]:
                result.update(iou_heldout_ok=result["iou_heldout"], wmae_heldout_ok=result["wmae_heldout"])
            if args.fill and hidden:  # physics track: the predicted hidden values inside the ground-truth program
                meta = metas[s["scene_id"]]
                truth_code = Path(meta["program_path"]).read_text(encoding="utf-8")
                if fill_scope_complete(meta, truth_code):
                    filled, fallback_share = fill_program(meta, truth_code, s["_physics"], hidden, fallback[s["scene_id"]])
                    rollout = (simulate(meta, filled, args.sim_timeout, out / "videos_fill") if filled is not None
                               else {**FAILED_ROLLOUT, "sim_success": 0.0, "sim_error": "missing_fields"})
                    result.update({f"fill_{k}": v for k, v in rollout.items()}, fill_fallback=fallback_share)
            return result

        with ThreadPoolExecutor(max_workers=args.sim_workers) as pool:
            for s, result in zip(scores, pool.map(run, scores)):
                s.update(result)
        side_by_side(out, metas, scores, "", out / "videos")
        side_by_side(out, metas, scores, "fill_", out / "videos_fill")
    for s in scores:
        s.pop("_codes")
        s.pop("_physics")

    summary = {
        "pred": str(args.pred),
        "config": config,
        "n": len(scores),
        "metrics": {m: bootstrap(scores, m, args.bootstrap) for m in HEADLINE},
        "by_variant": {
            variant: {m: mean([s.get(m) for s in scores if s["variant"] == variant]) for m in HEADLINE}
            for variant in sorted({s["variant"] for s in scores})
        },
        "sensitivity": sensitivity(scores),
    }
    write_jsonl(out / "scores.jsonl", scores)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(tables([summary]))
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_name or out.name, job_type="eval",
                         config={"pred": str(args.pred), "split": args.split, "simulate": args.simulate})
        for metric, entry in summary["metrics"].items():
            if entry.get("mean") is not None:
                run.summary[f"test/{metric}"] = entry["mean"]
                run.summary[f"test/{metric}_ci95"] = entry["ci95"]
        run.summary.update({f"test/{k}": v for k, v in summary["sensitivity"].items()})
        run.log({"test/scores": wandb.Table(dataframe=__import__("pandas").DataFrame(scores))})
        run.finish()


def markdown(summaries: list[dict[str, Any]], names: list[str], metrics: tuple[str, ...]) -> str:
    def cell(entry: dict[str, Any] | None) -> str:
        if not entry or entry.get("mean") is None:
            return "-"
        lo, hi = entry["ci95"]
        return f"{entry['mean']:.3f} [{lo:.3f}, {hi:.3f}] (n={entry['n']})"
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {m} | " + " | ".join(cell(s["metrics"].get(m)) for s in summaries) + " |" for m in metrics]
    return "\n".join(lines)


def tables(summaries: list[dict[str, Any]], names: list[str] | None = None, everything: bool = False) -> str:
    names = names or [Path(s["pred"]).parent.name for s in summaries]
    if everything:
        return markdown(summaries, names, HEADLINE)
    return ("Physics track (inferred values; filled rollout inside the ground-truth program)\n\n"
            + markdown(summaries, names, PHYSICS_CORE)
            + "\n\nCode track (rollout of the model's own program)\n\n" + markdown(summaries, names, CODE_CORE))


def compare(args: argparse.Namespace) -> None:
    summaries = [json.loads(Path(p).read_text()) for p in args.summaries]
    print(tables(summaries, [Path(p).parent.name for p in args.summaries], args.all))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="run the model on a swift test jsonl")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter", help="LoRA checkpoint dir; omit for the zero-shot base model")
    p.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--num-samples", type=int, default=1, help="> 1: sampled responses for best-of-N (greedy otherwise)")
    p.add_argument("--temperature", type=float, default=0.7, help="used only with --num-samples > 1")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--physics-only", action="store_true", help="stop after the </physics> block (no program)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=generate)

    p = sub.add_parser("oracle", help="ground-truth responses as predictions (pipeline check / simulation ceiling)")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=oracle)

    p = sub.add_parser("baseline", help="train-set mode per field (blind physics baseline)")
    p.add_argument("--data", type=Path, default=Path("data/sft_v1"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=baseline)

    p = sub.add_parser("score", help="physics-track and code-track metrics with family-bootstrap CIs")
    p.add_argument("--data", type=Path, default=Path("data/sft_v1"))
    p.add_argument("--split", default="test")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--config", choices=tuple(HIDDEN), help="which quantities were hidden (default: the config named in --pred)")
    p.add_argument("--simulate", action="store_true", help="execute generated programs (untrusted code: use a container)")
    p.add_argument("--fill", action="store_true", help="with --simulate: also run the ground-truth program filled with the predicted hidden values")
    p.add_argument("--best-of-n", action="store_true", help="with --simulate: run the sampled responses in order, keep the first that runs")
    p.add_argument("--sim-timeout", type=int, default=900)
    p.add_argument("--sim-workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--wandb-project", help="also log the summary to this wandb project (e.g. pixels2physics)")
    p.add_argument("--wandb-name", help="wandb run name (default: output dir name)")
    p.set_defaults(func=score)

    p = sub.add_parser("compare", help="markdown table across summary.json files")
    p.add_argument("summaries", nargs="+")
    p.add_argument("--all", action="store_true", help="every metric instead of the two core tables")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
