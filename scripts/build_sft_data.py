"""Build ms-swift SFT datasets (C0-C3) from the SFT-2000 split.

Every record pairs N video frames (first 70% of the clip) with a prompt whose
auxiliary information depends on the experiment config, and a target of
`<physics>` JSON + a complete Taichi program. The target is identical across
configs, so only the given information changes between experiments.

    python scripts/build_sft_data.py --data-root data/SFT-2000 --out data/sft_v1

Outputs (all paths inside JSONL are absolute, so build on the training machine):
    frames/<scene_id>/{rgb,gray}_XX.jpg
    swift/<config>/{train,validation,test}.jsonl     ms-swift messages + images
    swift/controls/<config>__<control>/test.jsonl    blank / shuffled / swapped / gray
    meta/{train,validation,test}.jsonl               ground truth for evaluate.py
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "validation", "test")
GROUPS = ("material", "geometry", "motion")
CONFIGS = {
    "c0_video_only": (),
    "c1_full_aux": ("material", "geometry", "motion"),
    "c2_no_position": ("material", "motion"),
    "c3_no_material": ("geometry", "motion"),
}
CONTROLS = ("blank", "shuffled", "swapped", "gray")
OBSERVED_FRACTION = 0.70

SYSTEM_PROMPT = (
    "You are a computational physicist. You watch frames of a 2-D MLS-MPM simulation "
    "and reconstruct it as an executable Python Taichi program."
)

# Leaf classification. Keys come from free-form YAML, so rules match key names.
RENDER_SECTIONS = {"visualization", "rendering", "render", "visual"}
MATERIAL_SECTIONS = {"physics_properties", "material_properties", "material", "plasticity_properties", "plasticity", "fluid_properties"}
RENDER_KEY = re.compile(r"colou?r|particle_radius|line_width|palette|alpha|draw")
MATERIAL_KEY = re.compile(
    r"constitutive|elasticity|density|^rho$|young|poisson|modulus|^e$|^nu$|lame|lambda|^mu(_0)?$|hardening|yield|"
    r"plastic|friction|viscosity|cohesion|theta|stiffness|bulk|restitution|damping"
)
MOTION_KEY = re.compile(r"veloc|speed|omega|angular|amplitude|frequency|phase|period|spin|accel|motion|emission_rate|flow_rate")
COUNT_KEY = re.compile(r"count|^num_|particles|jitter|std_dev|fraction|layers|distribution")
SHAPE_DETAIL_KEY = re.compile(r"segments|modulation|inner_amplitude|outer_amplitude")
TIME_KEY = re.compile(r"time|frames")
GEOMETRY_KEY = re.compile(
    r"corner|cent(er|re)|lower|upper|min|max|region|position|^pos$|radius|size|width|height|length|endpoint|start|end|"
    r"pivot|extent|half|offset|initial_[xyz]$|span|thickness|vertices|points|^[xyz]$|bounds|top|bottom|left|right|"
    r"origin|location|angle|orientation|scale"
)
CONTEXT_OUTPUT_KEY = re.compile(r"fps|framerate|frame_rate|background|resolution")


def leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [item for key in value for item in leaves(value[key], f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return [item for i, child in enumerate(value) for item in leaves(child, f"{prefix}[{i}]")]
    return [(prefix, value)]


def is_numeric(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(is_numeric(v) for v in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def classify(path: str, value: Any) -> str:
    segments = [re.sub(r"(\[\d+\])+$", "", s).lower() for s in path.split(".")]
    key = segments[-1]
    if RENDER_SECTIONS & set(segments) or RENDER_KEY.search(key):
        return "render"
    if MATERIAL_SECTIONS & set(segments[:-1]) or MATERIAL_KEY.search(key):
        return "material"
    if COUNT_KEY.search(key):
        return "structure"
    if SHAPE_DETAIL_KEY.search(key):
        return "geometry" if is_numeric(value) else "structure"
    if MOTION_KEY.search(key) or TIME_KEY.search(key):
        return "motion"
    if GEOMETRY_KEY.search(key) or is_numeric(value):  # any other number is layout (p1, x_range, floor_y, ...)
        return "geometry"
    return "structure"


def clean_number(value: Any) -> Any:
    """Round floats to 4 significant digits so targets have one canonical spelling."""
    if isinstance(value, (bool, int)) or not isinstance(value, float):
        return [clean_number(v) for v in value] if isinstance(value, list) else value
    if not math.isfinite(value):
        return str(value)
    return float(f"{float(value):.4g}")


def dumps(value: Any) -> str:
    """Compact JSON with numbers in %.4g form, so magnitudes show up as exponent tokens (5e+03)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {dumps(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(dumps(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return json.dumps(value, ensure_ascii=False)


def extract_scene(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split a YAML scene into always-given context and per-object grouped fields."""
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    context = {
        "simulation": clean_number(config.get("simulation", {})),
        "environment": clean_number(config.get("environment", {})),
        "output": {k: clean_number(v) for k, v in output.items() if CONTEXT_OUTPUT_KEY.search(k.lower())},
    }
    objects = []
    raw_objects = [o for o in config.get("objects") or [] if isinstance(o, dict)]
    for index, raw in enumerate(raw_objects, start=1):
        obj: dict[str, Any] = {"id": f"obj{index}", "role": raw.get("object_type") or raw.get("type") or "unknown"}
        obj.update({group: {} for group in ("structure", "render", *GROUPS)})
        body = {k: v for k, v in raw.items() if k not in {"name", "object_type", "type", "description"}}
        for path, value in leaves(body):
            obj[classify(path, value)][path] = clean_number(value)
        objects.append(obj)
    return context, objects


def physics_target(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": o["id"], **{g: o[g] for g in GROUPS if o[g]}} for o in objects]


def given_objects(objects: list[dict[str, Any]], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    keep = ("structure", "render", *groups)
    return [{"id": o["id"], "role": o["role"], **{g: o[g] for g in keep if o[g]}} for o in objects]


def first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else None


def seconds_per_frame(simulation: dict[str, Any]) -> float | None:
    """Physical simulation time per rendered frame (dt x substeps), tolerant to free-form YAML."""
    if isinstance(simulation.get("target_time_per_frame"), (int, float)):
        return float(simulation["target_time_per_frame"])
    dt = simulation.get("time_step_dt") or simulation.get("dt")
    dt = float(dt) if isinstance(dt, (int, float)) else None
    for key in ("substeps_per_frame", "steps_per_frame", "pbmpm_substeps_per_frame"):
        value = simulation.get(key)
        if isinstance(value, (int, float)) and dt:
            return dt * float(value)
        if isinstance(value, str) and dt:
            if "/ dt" in value or "/dt" in value:  # e.g. "≈1e-2 / dt": the number is the frame time itself
                seconds = first_number(value)
                return seconds if seconds else None
            numbers = re.findall(r"\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            return dt * float(numbers[-1]) if numbers else None
    return None


def frame_indices(total: int, n: int) -> list[int]:
    end = math.ceil(total * OBSERVED_FRACTION)
    return [math.floor((i + 0.5) * end / n) for i in range(n)]


def extract_frames(video: Path, indices: list[int], out_dir: Path, size: int, gray: bool) -> None:
    variants = {"rgb": f"scale={size}:{size}:flags=area"}
    if gray:
        variants["gray"] = f"scale={size}:{size}:flags=area,format=gray"
    select = "select='" + "+".join(f"eq(n\\,{i})" for i in indices) + "'"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, scale in variants.items():
        if all((out_dir / f"{name}_{k:02d}.jpg").is_file() for k in range(len(indices))):
            continue
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vf", f"{select},{scale}", "-vsync", "vfr",
             "-q:v", "2", "-start_number", "0", str(out_dir / f"{name}_%02d.jpg")],
            check=True, timeout=300,
        )
        written = sorted(out_dir.glob(f"{name}_*.jpg"))
        if len(written) != len(indices):
            raise RuntimeError(f"{video}: expected {len(indices)} frames, ffmpeg wrote {len(written)}")


def user_prompt(meta: dict[str, Any], context: dict[str, Any], objects: list[dict[str, Any]], config: str, output: str) -> str:
    spf = meta["seconds_per_frame"]
    lines = []
    for k, index in enumerate(meta["frame_indices"]):
        stamp = f"t = {index * spf:.4g} s" if spf else f"video frame {index}"
        lines.append(f"Frame {k + 1} ({stamp}): <image>")
    given, hidden = CONFIGS[config], [g for g in GROUPS if g not in CONFIGS[config]]
    lines += [
        "",
        f"The frames are evenly spaced over the first {OBSERVED_FRACTION:.0%} of the simulation"
        + (f"; one rendered frame = {spf:.4g} s of simulated time." if spf else "."),
        "Scene context (solver, environment, output):",
        dumps(context),
        "Objects" + (f" with given {', '.join(given)} properties:" if given else " (roles, structure and rendering only):"),
        "\n".join(dumps(o) for o in objects),
        "",
    ]
    task = f"Infer the {', '.join(hidden)} properties of every object from the frames. " if hidden else ""
    if output == "physics_code":
        task += (
            "First output a <physics></physics> block containing a JSON list with the material, geometry and motion "
            "fields of every object id, then output one complete executable Python Taichi MLS-MPM program in a "
            "```python block that reproduces the whole simulation."
        )
    else:
        task += "Output one complete executable Python Taichi MLS-MPM program in a ```python block that reproduces the whole simulation."
    return "\n".join(lines) + task


def assistant_response(physics: list[dict[str, Any]], code: str, output: str) -> str:
    block = f"```python\n{code.strip()}\n```"
    if output == "code":
        return block
    return "<physics>\n[\n" + ",\n".join(dumps(o) for o in physics) + "\n]\n</physics>\n" + block


def swift_row(meta: dict[str, Any], prompt: str, response: str | None, images: list[str]) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return {"messages": messages, "images": images, "scene_id": meta["scene_id"]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/SFT-2000"))
    parser.add_argument("--out", type=Path, default=Path("data/sft_v1"))
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=384, help="square resize; multiple of 32 for Qwen3-VL")
    parser.add_argument("--output-format", choices=("physics_code", "code"), default="physics_code")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-controls", action="store_true", help="skip test-time control sets")
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    data_root, out = args.data_root.resolve(), args.out.resolve()
    raw_root = data_root / "raw"
    records = [json.loads(line) for line in (data_root / "canonical_records.jsonl").read_text().splitlines() if line]

    metas, scenes = [], {}
    for record in records:
        scene_dir = raw_root / Path(record["program_path"]).parent
        config = yaml.safe_load((scene_dir / "scene.yaml").read_text(encoding="utf-8"))
        context, objects = extract_scene(config)
        total = int(record["observation"]["video_metadata"]["frame_count"])
        meta = {
            "scene_id": record["scene_id"],
            "parent_id": record["parent_id"],
            "variant": record["variant"],
            "split": record["split"],
            "video_path": str(scene_dir / "video.mp4"),
            "program_path": str(scene_dir / "program.py"),
            "frame_count": total,
            "frame_indices": frame_indices(total, args.num_frames),
            "heldout_start_frame": math.ceil(total * OBSERVED_FRACTION),
            "seconds_per_frame": seconds_per_frame(config.get("simulation") or {}),
            "physics": physics_target(objects),
            "roles": [o["role"] for o in objects],
        }
        metas.append(meta)
        scenes[meta["scene_id"]] = (context, objects, (scene_dir / "program.py").read_text(encoding="utf-8"))

    frames_root = out / "frames"
    with_gray = not args.no_controls

    def extract(meta: dict[str, Any]) -> None:
        gray = with_gray and meta["split"] == "test"
        extract_frames(Path(meta["video_path"]), meta["frame_indices"], frames_root / meta["scene_id"], args.frame_size, gray)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(extract, metas))

    def images(scene_id: str, kind: str = "rgb") -> list[str]:
        return [str(frames_root / scene_id / f"{kind}_{k:02d}.jpg") for k in range(args.num_frames)]

    for split in SPLITS:
        split_metas = [m for m in metas if m["split"] == split]
        write_jsonl(out / "meta" / f"{split}.jsonl", split_metas)
        for config in CONFIGS:
            rows = []
            for meta in split_metas:
                context, objects, code = scenes[meta["scene_id"]]
                prompt = user_prompt(meta, context, given_objects(objects, CONFIGS[config]), config, args.output_format)
                rows.append(swift_row(meta, prompt, assistant_response(meta["physics"], code, args.output_format), images(meta["scene_id"])))
            write_jsonl(out / "swift" / config / f"{split}.jsonl", rows)

    if not args.no_controls:
        rng = random.Random(args.seed)
        test = [m for m in metas if m["split"] == "test"]
        blank = frames_root / "blank.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={args.frame_size}x{args.frame_size}",
                        "-frames:v", "1", str(blank)], check=True)
        # Swap partner: a test scene from a different family, chosen once and shared by all configs.
        partners = {m["scene_id"]: rng.choice([o["scene_id"] for o in test if o["parent_id"] != m["parent_id"]]) for m in test}
        orders = {m["scene_id"]: rng.sample(range(args.num_frames), args.num_frames) for m in test}
        control_images = {
            "blank": lambda sid: [str(blank)] * args.num_frames,
            "shuffled": lambda sid: [images(sid)[k] for k in orders[sid]],
            "swapped": lambda sid: images(partners[sid]),
            "gray": lambda sid: images(sid, "gray"),
        }
        for config in CONFIGS:
            for control in CONTROLS:
                rows = []
                for meta in test:
                    context, objects, code = scenes[meta["scene_id"]]
                    prompt = user_prompt(meta, context, given_objects(objects, CONFIGS[config]), config, args.output_format)
                    rows.append(swift_row(meta, prompt, None, control_images[control](meta["scene_id"])))
                write_jsonl(out / "swift" / "controls" / f"{config}__{control}" / "test.jsonl", rows)

    manifest = {
        "data_root": str(data_root),
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "output_format": args.output_format,
        "configs": {name: list(groups) for name, groups in CONFIGS.items()},
        "controls": [] if args.no_controls else list(CONTROLS),
        "split_counts": dict(Counter(m["split"] for m in metas)),
        "missing_seconds_per_frame": sum(m["seconds_per_frame"] is None for m in metas),
        "field_counts": {g: sum(len(o.get(g, {})) for m in metas for o in m["physics"]) for g in GROUPS},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
