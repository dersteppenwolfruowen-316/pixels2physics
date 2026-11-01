"""Generate predictions and score them at four levels.

    # 1. predictions (vLLM; a LoRA adapter is merged first)
    python scripts/evaluate.py generate --dataset data/sft_v1/swift/c3_no_material/test.jsonl \
        --adapter runs/c3_no_material__weighted/vX/checkpoint-best --out runs/eval/c3/pred.jsonl
    # reference predictions: ground truth (simulation ceiling) and train-set mode (blind baseline)
    python scripts/evaluate.py oracle   --dataset data/sft_v1/swift/c3_no_material/test.jsonl --out runs/eval/oracle/pred.jsonl
    python scripts/evaluate.py baseline --data data/sft_v1 --out runs/eval/mode/pred.jsonl
    # 2. scores (+ --simulate executes generated programs: run inside a container on a GPU box)
    python scripts/evaluate.py score --data data/sft_v1 --pred runs/eval/c3/pred.jsonl --simulate
    # 3. one table across runs
    python scripts/evaluate.py compare runs/eval/*/summary.json

Levels:
    L0 validity     physics JSON parses, code parses, JSON numbers appear in code
    L1 parameters   per object id and YAML field path: constitutive model accuracy, log10 error of
                    density / Young's modulus, Poisson error, geometry & motion accuracy within tolerance, key recall
    L2 simulation   run the program, foreground-mask IoU against the GT video on the observed 70%
                    and on the held-out last 30% (never shown to the model)
    L3 sensitivity  within a family: does the predicted material change when the true one does?
                    (controls: score the swift/controls/* predictions with the same command)
All headline numbers carry 95% bootstrap CIs resampled over families (parent_id).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
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
    "geometry_acc_tol", "geometry_mae_unit", "geometry_key_recall", "motion_acc_tol", "motion_key_recall",
    "sim_success", "iou_observed", "iou_heldout",
)


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
    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0, seed=0)
    responses = engine.infer(requests, config, use_tqdm=True)
    write_jsonl(args.out, [{"scene_id": row["scene_id"], "response": r.choices[0].message.content} for row, r in zip(rows, responses)])


def oracle(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.dataset)
    write_jsonl(args.out, [{"scene_id": row["scene_id"], "response": row["messages"][-1]["content"]} for row in rows])


def baseline(args: argparse.Namespace) -> None:
    """Blind baseline: guess the constitutive model as the train-set mode for the object's role, then every
    other field as the train-set mode among objects with that role and that guessed model."""
    train, test = read_jsonl(args.data / "meta" / "train.jsonl"), read_jsonl(args.data / "meta" / f"{args.split}.jsonl")
    pools: dict[tuple, Counter] = defaultdict(Counter)
    for meta in train:
        for role, obj in zip(meta["roles"], meta["physics"]):
            cm = constitutive_model(obj)
            for group in GROUPS:
                for path, value in obj.get(group, {}).items():
                    for key in ((role, cm, group, path), (role, group, path), (group, path)):
                        pools[key][json.dumps(value)] += 1
    rows = []
    for meta in test:
        predicted = []
        for role, obj in zip(meta["roles"], meta["physics"]):
            cm_path = next((p for p in obj.get("material", {}) if p.endswith("constitutive_model")), None)
            cm_pool = pools[(role, "material", cm_path)] if cm_path else None
            cm = json.loads(cm_pool.most_common(1)[0][0]) if cm_pool else None
            guess = {"id": obj["id"]}
            for group in GROUPS:
                fields = {}
                for path in obj.get(group, {}):
                    pool = next((pools[k] for k in ((role, cm, group, path), (role, group, path), (group, path)) if pools[k]), None)
                    if pool:
                        fields[path] = json.loads(pool.most_common(1)[0][0])
                guess[group] = fields
            predicted.append(guess)
        rows.append({"scene_id": meta["scene_id"], "response": f"<physics>\n{json.dumps(predicted)}\n</physics>"})
    write_jsonl(args.out, rows)


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
            item["abs_err"] = mean([abs(a - b) for a, b in zip(t, p)])
            item["unit_abs_err"] = mean([abs(a - b) for a, b in zip(t, p) if abs(a) <= 1.5])  # domain coordinates
            if len(t) == 1 and t[0] > 0 and p[0] > 0:
                item["log10_err"] = abs(math.log10(p[0] / t[0]))
        else:
            item["exact"] = item["tol_ok"] = guess == value
        items.append(item)
    return items


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
        **{f"{g}_acc_tol": mean([float(i["tol_ok"]) for i in items[g]]) for g in ("geometry", "motion")},
        **{f"{g}_key_recall": mean([float(i["present"]) for i in items[g]]) for g in ("geometry", "motion")},
    }


def json_code_consistency(physics: list[dict[str, Any]] | None, code: str | None) -> float | None:
    if not physics or not code:
        return None
    wanted = {float(f"{float(n):.4g}") for n in NUMBER.findall(json.dumps(physics))} - {0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}
    found = {float(f"{float(n):.4g}") for n in NUMBER.findall(code)}
    return len(wanted & found) / len(wanted) if wanted else None


# ----------------------------------------------------------------------------- L2

def decode_gray(video: Path, size: int = 128):
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={size}:{size}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size).astype(np.int16)


def foreground(frames):
    import numpy as np

    border = np.concatenate([frames[0, 0], frames[0, -1], frames[0, :, 0], frames[0, :, -1]])
    return np.abs(frames - np.median(border)) > 30


def mask_iou(truth, predicted, indices: range) -> float:
    scores = []
    for i in indices:
        if i >= len(predicted):
            scores.append(0.0)  # the program stopped early: missing frames count as misses
            continue
        union = (truth[i] | predicted[i]).sum()
        scores.append(float((truth[i] & predicted[i]).sum() / union) if union else 1.0)
    return sum(scores) / len(scores)


def simulate(meta: dict[str, Any], code: str, timeout: int, video_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="p2p_sim_") as tmp:
        (Path(tmp) / "program.py").write_text(code, encoding="utf-8")
        try:
            subprocess.run([sys.executable, "program.py"], cwd=tmp, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"sim_success": 0.0, "sim_error": "timeout"}
        videos = sorted(Path(tmp).rglob("*.mp4"), key=lambda p: p.stat().st_size)
        if not videos:
            return {"sim_success": 0.0, "sim_error": "no_video"}
        video_dir.mkdir(parents=True, exist_ok=True)
        kept = video_dir / f"{meta['scene_id']}.mp4"
        shutil.copy(videos[-1], kept)
    truth, predicted = foreground(decode_gray(Path(meta["video_path"]))), foreground(decode_gray(kept))
    held = meta["heldout_start_frame"]
    return {
        "sim_success": 1.0,
        "iou_observed": mask_iou(truth, predicted, range(0, held, 3)),
        "iou_heldout": mask_iou(truth, predicted, range(held, len(truth), 3)),
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


def score(args: argparse.Namespace) -> None:
    metas = {m["scene_id"]: m for m in read_jsonl(args.data / "meta" / f"{args.split}.jsonl")}
    predictions = read_jsonl(args.pred)
    out = args.out_dir or args.pred.parent
    scores = []
    for row in predictions:
        meta = metas[row["scene_id"]]
        physics, code = parse_response(row["response"])
        try:
            parses = code is not None and ast.parse(code) is not None
        except SyntaxError:
            parses = False
        scores.append({
            "scene_id": meta["scene_id"], "parent_id": meta["parent_id"], "variant": meta["variant"],
            "physics_valid": float(physics is not None),
            "code_parses": float(parses),
            "json_code_consistency": json_code_consistency(physics, code),
            **score_physics(meta["physics"], physics),
            "roles": meta["roles"],
            "true_material": material_signature(meta["physics"]),
            "pred_material": material_signature(physics),
            "_code": code if parses else None,
        })
    if args.simulate:
        def run(s: dict[str, Any]) -> dict[str, Any]:
            if s["_code"] is None:
                return {"sim_success": 0.0, "sim_error": "no_code"}
            return simulate(metas[s["scene_id"]], s["_code"], args.sim_timeout, out / "videos")

        with ThreadPoolExecutor(max_workers=args.sim_workers) as pool:
            for s, result in zip(scores, pool.map(run, scores)):
                s.update(result)
    for s in scores:
        s.pop("_code")

    summary = {
        "pred": str(args.pred),
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
    print(markdown([summary]))
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


def markdown(summaries: list[dict[str, Any]], names: list[str] | None = None) -> str:
    names = names or [Path(s["pred"]).parent.name for s in summaries]
    def cell(entry: dict[str, Any]) -> str:
        if entry.get("mean") is None:
            return "-"
        lo, hi = entry["ci95"]
        return f"{entry['mean']:.3f} [{lo:.3f}, {hi:.3f}]"
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {m} | " + " | ".join(cell(s["metrics"][m]) for s in summaries) + " |" for m in HEADLINE]
    for key in ("material_change_detected", "material_stable_under_perturbation"):
        values = [s["sensitivity"][key] for s in summaries]
        lines.append(f"| {key} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in values) + " |")
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> None:
    summaries = [json.loads(Path(p).read_text()) for p in args.summaries]
    print(markdown(summaries, [Path(p).parent.name for p in args.summaries]))


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

    p = sub.add_parser("score", help="L0-L3 metrics with family-bootstrap CIs")
    p.add_argument("--data", type=Path, default=Path("data/sft_v1"))
    p.add_argument("--split", default="test")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--simulate", action="store_true", help="execute generated programs (untrusted code: use a container)")
    p.add_argument("--sim-timeout", type=int, default=900)
    p.add_argument("--sim-workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--wandb-project", help="also log the summary to this wandb project (e.g. pixels2physics)")
    p.add_argument("--wandb-name", help="wandb run name (default: output dir name)")
    p.set_defaults(func=score)

    p = sub.add_parser("compare", help="markdown table across summary.json files")
    p.add_argument("summaries", nargs="+")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
