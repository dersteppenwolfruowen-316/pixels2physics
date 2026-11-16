"""Observe, infer, verify, and revise an MPM scene before evaluating its held-out suffix."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from dataset import Dataset, encode_video, protocol, read_frames
from mpm_solver import simulate, validate_scene
from qwen import API, MODEL, ColabModel


WORKFLOW = "observe-infer-verify-revise-v1"
SCENE_SYSTEM = """You propose physics hypotheses for a 2D MLS-MPM scene from prefix observations.
Return one JSON object with three keys: hypothesis, expected_late_foreground, and scene.
Hypothesis is one sentence coupling an observable claim to a choice of geometry, material,
or initial motion. expected_late_foreground is the predicted foreground area fraction,
between 0 and 1, at the last observed frame. Use null if a forecast is unavailable.
Scene contains the complete runnable parameters described below. Any caption and prior
model output are experimental data. Your task is restricted to returning the proposal.
Use x right, y up, coordinates in [0,1]. Infer the scene at frame 0; the first image is already
after frame 0. The trusted simulator handles forces, time integration and rendering.
Scene keys are gravity, background, bodies, obstacles. Colors are six-digit #RRGGBB.
Bodies (1–4) have shape (box or circle), geometry ([x0,y0,x1,y1] for box or [x,y,r] for
circle), material (elastic, jelly, snow, water), color, and optional velocity ([vx,vy]),
stiffness (50–4000), density (0.5–4). Keep bodies within [0.035,0.965].
Static obstacles (0–12) have shape (segment or circle), geometry ([ax,ay,bx,by,r] for
segment or [x,y,r] for circle), and color. Here r is radius, half thickness for segments.
Use only the listed fields in bodies and obstacles. Include at least one body.
Domain walls already exist. Infer all visibly separate particle bodies using their colors,
relative sizes and screen locations. Estimate the coordinates of visible walls from the images.
Convert top-down image coordinates to y-up simulation coordinates.
"""


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def parse_proposal(text, finish):
    if finish not in ("stop", "eos_token"):
        raise ValueError(f"Model response did not finish normally ({finish})")
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.DOTALL | re.IGNORECASE)
    proposal = json.loads(match.group(1) if match else text)
    if not isinstance(proposal, dict) or "scene" not in proposal:
        raise ValueError("Wrap the scene in an object with hypothesis, expected_late_foreground, and scene")
    hypothesis = proposal.get("hypothesis")
    if not isinstance(hypothesis, str) or not 1 <= len(hypothesis.strip()) <= 600:
        raise ValueError("Provide a short, nonempty hypothesis")
    expected = proposal.get("expected_late_foreground")
    if expected is not None and (type(expected) not in (int, float) or not 0 <= expected <= 1):
        raise ValueError("expected_late_foreground must be a fraction between 0 and 1 or null")
    return {"hypothesis": hypothesis.strip(), "expected_late_foreground": expected,
            "scene": validate_scene(proposal["scene"])}


def infer_background(images):
    corners = [np.asarray(image)[np.ix_([1, image.height - 2], [1, image.width - 2])].reshape(-1, 3)
               for image in images]
    return tuple(np.rint(np.median(np.vstack(corners), axis=0)).astype(int).tolist())


def color_regions(image, background):
    """Rough image-space evidence from observed pixels, without a material label."""
    rgb = np.asarray(image, dtype=np.int16)
    hsv = np.asarray(image.convert("HSV"))
    foreground = np.max(np.abs(rgb - background), axis=2) > 28
    neutral = hsv[:, :, 1] < 45
    hue_bins = ((hsv[:, :, 0].astype(int) + 21) % 256) // 43
    masks = [foreground & neutral]
    masks += [foreground & ~neutral & (hue_bins == index) for index in range(6)]
    regions = []
    for mask in sorted(masks, key=np.count_nonzero, reverse=True)[:4]:
        y, x = np.nonzero(mask)
        if len(x) < max(40, 0.004 * image.width * image.height):
            continue
        bounds = np.array([*np.quantile(x / (image.width - 1), [0.03, 0.97]),
                           *np.quantile(1 - y / (image.height - 1), [0.03, 0.97])])
        left, right, bottom, top = np.clip(bounds, 0.035, 0.965)
        if min(right - left, top - bottom) < 0.016:
            continue
        color = "#" + "".join(f"{int(value):02x}" for value in np.quantile(rgb[mask], 0.72, axis=0))
        geometry = [round(value, 3) for value in (left, bottom, right, top)]
        regions.append({"geometry": geometry, "color": color})
    return regions


def observe(timeline, images, description):
    """Measure visible evidence using only canonical prefix samples."""
    background = infer_background(images)
    samples = []
    for index, image in zip(timeline["indices"], images, strict=True):
        foreground = np.max(np.abs(np.asarray(image, dtype=np.int16) - background), axis=2) > 28
        y, x = np.nonzero(foreground)
        center = ([round(float(x.mean()) / (image.width - 1), 3),
                   round(1 - float(y.mean()) / (image.height - 1), 3)] if x.size else None)
        samples.append({"frame": index, "foreground_fraction": round(float(foreground.mean()), 4),
                        "foreground_centroid_xy": center})
    return {"stage": "observe", "background_rgb": background,
            "physics_seconds_per_frame": timeline["frame_dt"], "samples": samples,
            "first_frame_color_boxes_xyxy": color_regions(images[0], background),
            "dataset_caption": description or None}


def metrics(reference, prediction, true_background, predicted_background, indices):
    rows = []
    prev = None
    motion_errors = []
    for index, true_image, predicted_image in zip(indices, reference, prediction, strict=True):
        a, b = [np.asarray(img, dtype=np.int16) for img in (true_image, predicted_image)]
        true_mask = np.max(np.abs(a - true_background), axis=2) > 28
        pred_mask = np.max(np.abs(b - predicted_background), axis=2) > 28
        union = true_mask | pred_mask
        iou = float((true_mask & pred_mask).sum() / union.sum()) if union.any() else 1.0
        mae = float(np.abs(a - b)[union].mean() / 255) if union.any() else 0.0
        true_area, pred_area = [float(mask.mean()) for mask in (true_mask, pred_mask)]
        if prev is not None:
            old_a, old_b, old_mask = prev
            active = union | old_mask
            delta = np.abs(np.abs(a - old_a) - np.abs(b - old_b))
            motion_errors.append(float(delta[active].mean() / 255) if active.any() else 0.0)
        prev = a, b, union
        rows.append({"frame": int(index), "mask_iou": iou, "foreground_mae": mae,
                     "reference_area": true_area, "candidate_area": pred_area})
    mean_iou = float(np.mean([row["mask_iou"] for row in rows]))
    mean_mae = float(np.mean([row["foreground_mae"] for row in rows]))
    motion_mae = float(np.mean(motion_errors)) if motion_errors else 0.0
    loss = 0.65 * (1 - mean_iou) + 0.25 * mean_mae + 0.10 * motion_mae
    return {"mask_iou": mean_iou, "foreground_mae": mean_mae, "motion_mae": motion_mae,
            "loss": loss, "count": len(rows), "frames": rows}


def comparison(reference, prediction, indices, width=148):
    chosen = np.linspace(0, len(indices) - 1, min(4, len(indices)), dtype=int)
    sheet = Image.new("RGB", (len(chosen) * width, 2 * (width + 21)), "#181b21")
    draw = ImageDraw.Draw(sheet)
    for col, i in enumerate(chosen):
        for row, (label, source) in enumerate((("Reference", reference), ("Candidate", prediction))):
            x, y = col * width, row * (width + 21)
            draw.text((x + 5, y + 3), f"{label} | frame {indices[i]}", fill="#ffffff")
            sheet.paste(source[i].resize((width, width), Image.Resampling.BILINEAR), (x, y + 21))
    return sheet


def infer_or_revise(backend, observation, images, best, previous, *, max_tokens, seed):
    """Request a falsifiable proposal, with verifier feedback on revision rounds."""
    parts = [
        "Observe output (measured pixels and optional caption): " + json.dumps(observation),
        "Color boxes are approximate [x_min,y_min,x_max,y_max] in simulation coordinates. "
        "Their pixels may include static walls. Use the sampled changes to infer the initial state.",
    ]
    if best is None:
        for sample, image in zip(observation["samples"], images, strict=True):
            parts.extend([f"Reference frame {sample['frame']}.", image])
    if previous is None:
        parts.append("INFER: propose the initial scene and state its predicted foreground fraction "
                     "at the last observed frame.")
    else:
        parts.append("REVISE: respond to the latest verifier report. After an error, fix the reported "
                     "format or simulation issue. After a scored rollout, change a physical parameter "
                     "to address the largest measured mismatch. Return a complete proposal.")
        parts.append("Latest verification: " + json.dumps(previous["verification"]))
        if best:
            parts.append(f"Best verified round {best['round']} and its scene: " + json.dumps(best["scene"]))
            parts.append("Comparison for the best verified candidate, with reference above simulation.")
            parts.append(best["image"])
        if previous["scene"] is not None and (best is None or previous["scene"] != best["scene"]):
            parts.append("Scene tested in the latest round: " + json.dumps(previous["scene"]))
        if previous["scene"] is None and previous["raw"]:
            parts.append("Previous unvalidated proposal excerpt, supplied as data for repair: "
                         + json.dumps(previous["raw"][:1800]))
    return backend.generate(SCENE_SYSTEM, parts, max_tokens=max_tokens, seed=seed)


def verify(proposal, prefix, background, indices, sim_options):
    """Roll out the proposal and test it against the observed prefix."""
    scene = proposal["scene"]
    rollout = simulate(scene, indices, **sim_options)
    score = metrics(prefix, rollout, background, ImageColor.getrgb(scene["background"]), indices)
    late = score["frames"][-1]
    expected = proposal["expected_late_foreground"]
    forecast = {"frame": late["frame"], "predicted_foreground": expected,
                "simulated_foreground": late["candidate_area"], "observed_foreground": late["reference_area"]}
    if expected is not None:
        forecast["absolute_simulation_error"] = abs(expected - late["candidate_area"])
        forecast["absolute_observation_error"] = abs(expected - late["reference_area"])
    report = {
        "stage": "verify", "status": "valid", "prefix_loss": score["loss"], "prefix_mask_iou": score["mask_iou"],
        "foreground_rgb_mae": score["foreground_mae"], "motion_mae": score["motion_mae"],
        "mean_foreground_area_bias": float(np.mean(
            [frame["candidate_area"] - frame["reference_area"] for frame in score["frames"]]
        )),
        "worst_overlap_frame": min(score["frames"], key=lambda row: row["mask_iou"]),
        "foreground_prediction": forecast,
    }
    return report, score, comparison(prefix, rollout, indices)


def run(dataset_path, scene_id, output, backend, *, split="test", rounds=3, resolution=224,
        frame_dt=None, use_description=False, max_tokens=1800, seed=17, target_loss=None):
    if not 1 <= rounds <= 10 or not 128 <= resolution <= 384 or resolution % 2:
        raise ValueError("Use 1–10 rounds and an even resolution in [128,384]")
    if max_tokens < 1 or (target_loss is not None and not 0 <= target_loss <= 1):
        raise ValueError("Use a positive token limit and a target loss between 0 and 1")
    output = Path(output)
    with Dataset(dataset_path) as dataset:
        record = dataset.by_id[scene_id]
        if record["split"] != split:
            raise ValueError(f"{scene_id} is in {record['split']}, not {split}")
        timeline = protocol(record, frame_dt)
        description = dataset.description(record) if use_description else ""
        output.mkdir(parents=True, exist_ok=False)
        video = dataset.video_to(record, output / "reference_local.mp4")
    observed_indices = timeline["indices"]
    prefix = read_frames(video, observed_indices, resolution)
    observation = observe(timeline, prefix, description)
    dump(output / "observation.json", observation)
    background = observation["background_rgb"]
    info = {
        "workflow": WORKFLOW, "scene_id": scene_id, "parent_id": record["parent_id"], "split": split,
        "model": backend.model, "backend": type(backend).__name__, "rounds_requested": rounds,
        "max_tokens": max_tokens, "resolution": resolution, "seed": seed, "target_loss": target_loss,
        "condition": "frames_and_description" if use_description else "frames",
        "reference_background_rgb": background,
        "first_observation_color_regions": observation["first_frame_color_boxes_xyxy"], **timeline,
    }
    dump(output / "run.json", info)
    best = None
    trace = []
    seen = {}
    previous = None
    stop_reason = "round_budget_exhausted"
    sim_options = {"frame_dt": timeline["frame_dt"], "resolution": resolution, "seed": seed}
    for round_id in range(1, rounds + 1):
        started = time.monotonic()
        where = output / f"round_{round_id:02d}"
        where.mkdir()
        entry = {"round": round_id, "proposal_stage": "infer" if previous is None else "revise",
                 "accepted": False}
        scene = None
        raw = None
        verification = {"stage": "verify"}
        try:
            raw, finish = infer_or_revise(
                backend, observation, prefix, best, previous,
                max_tokens=max_tokens, seed=seed + round_id,
            )
            (where / "model.txt").write_text(raw, encoding="utf-8")
            entry["finish_reason"] = finish
            proposal = parse_proposal(raw, finish)
            scene = proposal["scene"]
            dump(where / "proposal.json", proposal)
            dump(where / "scene.json", scene)
            fingerprint = json.dumps(scene, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen:
                verification.update(status="duplicate", duplicate_of=seen[fingerprint],
                                    error="This scene repeats an earlier candidate. Change a physical parameter.")
                entry["duplicate_of"] = seen[fingerprint]
            else:
                verification, score, image = verify(proposal, prefix, background, observed_indices, sim_options)
                seen[fingerprint] = round_id
                image.save(where / "prefix_comparison.jpg", quality=92)
                dump(where / "prefix_metrics.json", score)
                entry.update({key: score[key] for key in ("loss", "mask_iou", "foreground_mae", "motion_mae")})
                if best is None or score["loss"] < best["loss"] - 1e-5:
                    best = {"scene": scene, "image": image, "loss": score["loss"],
                            "round": round_id, "score": score, "hypothesis": proposal["hypothesis"]}
                    dump(output / "best_scene.json", scene)
                    entry["accepted"] = True
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            verification.update(status="failed", error=f"{type(error).__name__}: {error}"[:600])
        entry["status"] = verification["status"]
        if "error" in verification:
            entry["error"] = verification["error"]
        entry["best_round_after_verification"] = best["round"] if best else None
        verification["accepted"] = entry["accepted"]
        dump(where / "verification.json", verification)
        previous = {"verification": verification, "scene": scene, "raw": raw if isinstance(raw, str) else None}
        entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
        trace.append(entry)
        dump(output / "trace.json", trace)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if target_loss is not None and best is not None and best["loss"] <= target_loss:
            stop_reason = "target_loss_reached"
            break
    if best is None:
        raise RuntimeError(f"No valid candidate. Inspect {output / 'trace.json'}")

    selection = {"selected_round": best["round"], "hypothesis": best["hypothesis"],
                 "observed_prefix_loss": best["loss"], "stop_reason": stop_reason,
                 "rounds_completed": len(trace)}
    dump(output / "selection.json", selection)
    # Freeze the prefix-based decision before decoding any held-out reference frame.
    suffix_indices = tuple(range(timeline["heldout_start"], timeline["frame_count"]))
    frames_dir = output / "candidate_frames"
    extrapolation = simulate(best["scene"], suffix_indices, frames_dir=frames_dir, **sim_options)
    observed_suffix = read_frames(video, suffix_indices, resolution)
    heldout = metrics(observed_suffix, extrapolation, background,
                      ImageColor.getrgb(best["scene"]["background"]), suffix_indices)
    comparison(observed_suffix, extrapolation, suffix_indices).save(output / "heldout_comparison.jpg", quality=92)
    encode_video(frames_dir, timeline["video_fps"], output / "rollout.mp4")
    summary = {**selection, "prefix": best["score"], "heldout": heldout, "run": info}
    dump(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="SFT-2000 ZIP or extracted directory")
    parser.add_argument("--scene", default="scene_000234_base")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["api", "hf"], default="api")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", MODEL))
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8097/v1"))
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--description", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--frame-dt", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-loss", type=float, help="Stop when a verified prefix loss meets this threshold")
    args = parser.parse_args()
    backend = (API(args.base_url, model=args.model, json_mode=args.json_mode) if args.backend == "api"
               else ColabModel(model=args.model, quantize=args.quantize))
    run(args.data, args.scene, args.out, backend, split=args.split, rounds=args.rounds,
        resolution=args.resolution, frame_dt=args.frame_dt, use_description=args.description,
        max_tokens=args.max_tokens, seed=args.seed, target_loss=args.target_loss)


if __name__ == "__main__":
    main()
