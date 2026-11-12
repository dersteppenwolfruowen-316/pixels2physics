"""Staged agentic refinement: the SFT model rewrites its own program from verifier feedback (after SEIG).

Per scene:
  init     the model's best-of-8 samples (already generated) are simulated over the observed window; the runnable
           one that fits the 16 observed frames best is kept. If none runs, the scene starts at stage 0.
  stage 0  executable: feedback = the error (traceback tail, static hints); up to R0 rounds of k candidates; the
           first round with a runnable candidate keeps the best-fitting one and moves on. Never runnable = failure.
  stage 1  geometry  (frames 1-2)  \
  stage 2  motion    (frames 1-5)   } R rounds of k candidates each; feedback = comparison images + numbers;
  stage 3  force     (frames 1-16) /  greedy accept: the stage loss must drop and no earlier stage may get worse
                                       by more than GUARD.
Each round, every candidate is a complete response (<physics> + program) to a multi-turn conversation:
  [the SFT prompt] -> [the current best response] -> [feedback + this stage's focus].
The verifier sees only the 16 observed frames, never the held-out ones. All scenes advance in lock-step rounds so one
model call batches every scene's requests and the simulations of a round run in parallel.

  python agentic/loop.py run --name pilot --pilot 30 [--fake]      # agentic
  python agentic/loop.py resample --name pilot_resample --like pilot  # GPU: equal-budget samples, no feedback
  python agentic/loop.py select --name pilot_resample                 # CPU: simulate them, keep the best fit
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import feedback  # noqa: E402
import runner  # noqa: E402
from evaluate import parse_response, read_jsonl, write_jsonl  # noqa: E402
from feedback import STAGES  # noqa: E402

K, ROUNDS, ROUNDS_EXEC, GUARD = 4, 3, 3, 0.02


# ----------------------------------------------------------------------------- candidates and scenes

@dataclass
class Candidate:
    response: str
    code: str | None
    frames: np.ndarray | None = None  # the 16 observed-position frames of the rollout
    comp: feedback.Comparison | None = None
    error: str = ""
    detail: str = ""

    @property
    def runs(self) -> bool:
        return self.comp is not None

    def loss(self, stage: feedback.Stage) -> float:
        return self.comp.loss(stage) if self.comp else 1.0


@dataclass
class Scene:
    sid: str
    meta: dict[str, Any]
    row: dict[str, Any]
    observed: np.ndarray  # [16, S, S, 3]
    stage: int = 0  # 0 executable, 1..3 = STAGES[stage - 1], 4 = done
    round: int = 0
    best: Candidate | None = None
    generations: int = 0
    simulations: int = 0
    trace: list = field(default_factory=list)
    failed: bool = False

    @property
    def indices(self) -> list[int]:
        return self.meta["frame_indices"]

    @property
    def obs_end(self) -> int:
        return max(self.indices) + 1

    @property
    def done(self) -> bool:
        return self.stage > len(STAGES) or self.failed

    def times(self) -> list[str]:
        spf = self.meta["seconds_per_frame"]
        return [f"t = {i * spf:.3g} s" for i in self.indices]


def simulate(scene: Scene, response: str, timeout: int) -> Candidate:
    code = parse_response(response)[1]
    if code is None:
        return Candidate(response, None, error="no program", detail="Your answer had no ```python program block.")
    r = runner.run(code, scene.obs_end, timeout)
    if r.frames is None or r.error:
        return Candidate(response, code, error=r.error, detail=r.detail)
    frames = r.frames[scene.indices]
    return Candidate(response, code, frames, feedback.compare(scene.observed, r.frames, scene.indices))


# ----------------------------------------------------------------------------- prompts

OUTPUT = ("Answer in exactly the same format as before: first the <physics></physics> block with every object's "
          "material, geometry and motion fields and the scene gravity, then the complete corrected Python Taichi "
          "program in one ```python block. Write the whole program, not a diff.")


def request(scene: Scene, workdir: Path) -> tuple[list[dict], list[str]]:
    """Messages and images for the scene's next round."""
    messages = [m for m in scene.row["messages"] if m["role"] != "assistant"]
    images = list(scene.row["images"])
    best = scene.best
    frames_needed = (f"The observed video has {scene.meta['frame_count']} frames, one rendered frame every "
                     f"{scene.meta['seconds_per_frame']:.4g} s of simulated time; your program must render at least "
                     f"the first {scene.obs_end} of them (the frames you were shown are frames "
                     f"{', '.join(str(i + 1) for i in scene.indices[:3])}, ... of the video).")
    if scene.stage == 0:
        text = ("Your program does not work yet.\n```\n" + (best.detail or best.error).strip() + "\n```\n"
                + frames_needed + "\nFix the program so that it runs and renders these frames. " + OUTPUT)
    else:
        stage = STAGES[scene.stage - 1]
        parts = [f"Your program runs. Here is your simulation next to the observation ({stage.name} check)."]
        for p in stage.panels:
            path = workdir / "feedback" / f"{scene.sid}__s{scene.stage}r{scene.round}_f{p + 1}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            feedback.composite(scene.observed[p], best.frames[p], f"frame {p + 1}").save(path)
            parts.append(f"Frame {p + 1} ({scene.times()[p]}): <image>")
            images.append(str(path))
        parts.append(feedback.stage_text(best.comp, stage, scene.times()))
        history = [t for t in scene.trace if t["stage"] == scene.stage]
        if history:
            parts.append("Earlier attempts in this check: " + "; ".join(
                f"round {t['round'] + 1}: best candidate loss {t['best_candidate_loss']:.2f}, "
                + ("accepted" if t["accepted"] else "rejected (kept the previous program)") for t in history) + ".")
        parts.append(stage.focus + " " + OUTPUT)
        text = "\n".join(parts)
    messages = messages + [{"role": "assistant", "content": best.response}, {"role": "user", "content": text}]
    return messages, images


# ----------------------------------------------------------------------------- generators

class Generator:
    def generate(self, requests: list[tuple[str, list[dict], list[str], int]]) -> list[list[str]]:
        raise NotImplementedError


class FakeGenerator(Generator):
    """Local dry run without a GPU: answers with the scene's own best-of-8 samples, at random."""

    def __init__(self, samples: dict[str, list[str]], seed: int = 0):
        self.samples, self.rng = samples, random.Random(seed)

    def generate(self, requests):
        return [[self.rng.choice(self.samples[sid]) for _ in range(n)] for sid, _, _, n in requests]


class VllmGenerator(Generator):
    def __init__(self, adapter: str, model: str = "Qwen/Qwen3-VL-4B-Instruct", max_model_len: int = 32768,
                 temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 8192, gpu_util: float = 0.6):
        import subprocess

        from swift.infer_engine import VllmEngine

        merged = Path(f"{adapter.rstrip('/')}-merged")
        if not merged.is_dir():
            subprocess.run(["swift", "export", "--adapters", adapter, "--merge_lora", "true", "--use_hf", "true"], check=True)
        self.engine = VllmEngine(str(merged), use_hf=True, max_model_len=max_model_len, limit_mm_per_prompt={"image": 64},
                                 gpu_memory_utilization=gpu_util)
        self.sampling = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def generate(self, requests):
        from swift.infer_engine import InferRequest, RequestConfig

        out: list[list[str] | None] = [None] * len(requests)
        by_n = defaultdict(list)
        for i, (_, _, _, n) in enumerate(requests):
            by_n[n].append(i)
        for n, ids in by_n.items():
            config = RequestConfig(n=n, seed=None, **self.sampling)
            batch = [InferRequest(messages=requests[i][1], images=requests[i][2]) for i in ids]
            for i, r in zip(ids, self.engine.infer(batch, config, use_tqdm=True)):
                out[i] = [c.message.content for c in r.choices]
        return out


# ----------------------------------------------------------------------------- the loop

def load_scenes(args) -> list[Scene]:
    metas = {m["scene_id"]: m for m in read_jsonl(args.data / "meta" / "test.jsonl")}
    rows = {r["scene_id"]: r for r in read_jsonl(args.data / "swift" / "c4_scaffold" / "test.jsonl")}
    if args.scenes:
        ids = args.scenes
    else:  # stratified by family: round-robin over shuffled families until the pilot size
        rng = random.Random(args.seed)
        families = defaultdict(list)
        for sid in sorted(metas):
            families[metas[sid]["parent_id"]].append(sid)
        for v in families.values():
            rng.shuffle(v)
        order, ids = sorted(families), []
        while len(ids) < min(args.pilot, len(metas)):
            for fam in order:
                if families[fam] and len(ids) < args.pilot:
                    ids.append(families[fam].pop())
    scenes = []
    for sid in ids:
        observed = runner.decode_video(Path(metas[sid]["video_path"]))[metas[sid]["frame_indices"]]
        scenes.append(Scene(sid, metas[sid], rows[sid], observed))
    return scenes


def simulate_all(jobs: list[tuple[Scene, str]], args) -> list[Candidate]:
    with ThreadPoolExecutor(max_workers=args.sim_workers) as pool:
        return list(pool.map(lambda j: simulate(j[0], j[1], args.sim_timeout), jobs))


def pick(cands: list[Candidate], stage: feedback.Stage) -> Candidate | None:
    runnable = [c for c in cands if c.runs]
    return min(runnable, key=lambda c: c.loss(stage)) if runnable else None


def initialize(scenes: list[Scene], samples: dict[str, list[str]], args) -> None:
    jobs = [(s, r) for s in scenes for r in samples[s.sid]]
    results = simulate_all(jobs, args)
    for s in scenes:
        s.simulations += len(samples[s.sid])
    by_scene = defaultdict(list)
    for (s, _), c in zip(jobs, results):
        by_scene[s.sid].append(c)
    for s in scenes:
        cands = by_scene[s.sid]
        best = pick(cands, STAGES[-1])
        if best:
            s.best, s.stage = best, 1
        else:  # start the repair from the first sample that has a program
            s.best = next((c for c in cands if c.code), cands[0])
        s.trace.append({"stage": "init", "runnable": sum(c.runs for c in cands), "of": len(cands),
                        "errors": [c.error for c in cands if not c.runs],
                        "losses": {st.name: s.best.loss(st) for st in STAGES} if s.best.runs else None})


def step(scene: Scene, cands: list[Candidate]) -> None:
    """Apply one round's candidates to a scene."""
    entry: dict[str, Any] = {"stage": scene.stage, "round": scene.round,
                             "errors": [c.error for c in cands if not c.runs]}
    if scene.stage == 0:
        best = pick(cands, STAGES[-1])
        entry["accepted"] = best is not None
        entry["best_candidate_loss"] = best.loss(STAGES[-1]) if best else 1.0
        if best:
            scene.best, scene.stage, scene.round = best, 1, 0
        else:
            # Carry on from a candidate that fails differently, so the next round sees a new error.
            moved = next((c for c in cands if c.code and c.error != scene.best.error), None)
            if moved:
                scene.best = moved
            scene.round += 1
            if scene.round >= ROUNDS_EXEC:
                scene.failed = True
        scene.trace.append(entry)
        return
    stage = STAGES[scene.stage - 1]
    earlier = STAGES[: scene.stage - 1]
    valid = [c for c in cands if c.runs and all(c.loss(e) <= scene.best.loss(e) + GUARD for e in earlier)]
    best = pick(valid, stage)
    entry["candidate_losses"] = [round(c.loss(stage), 4) for c in cands]
    entry["best_candidate_loss"] = best.loss(stage) if best else 1.0
    entry["accepted"] = bool(best and best.loss(stage) < scene.best.loss(stage))
    if entry["accepted"]:
        scene.best = best
    entry["losses"] = {st.name: scene.best.loss(st) for st in STAGES}
    scene.trace.append(entry)
    scene.round += 1
    if scene.round >= ROUNDS:
        scene.stage, scene.round = scene.stage + 1, 0


def save(scenes: list[Scene], out: Path, init_responses: dict[str, str]) -> None:
    write_jsonl(out / "pred.jsonl", [{"scene_id": s.sid, "response": s.best.response} for s in scenes])
    write_jsonl(out / "init_pred.jsonl", [{"scene_id": s.sid, "response": init_responses[s.sid]} for s in scenes])
    write_jsonl(out / "trace.jsonl", [{"scene_id": s.sid, "final_stage": s.stage, "failed": s.failed,
                                       "generations": s.generations, "simulations": s.simulations, "trace": s.trace}
                                      for s in scenes])


def run(args) -> None:
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes(args)
    samples = {r["scene_id"]: r["responses"] for r in read_jsonl(args.init)}
    generator = FakeGenerator(samples) if args.fake else VllmGenerator(args.adapter, gpu_util=args.gpu_util)
    started = time.time()
    initialize(scenes, samples, args)
    init_responses = {s.sid: s.best.response for s in scenes}
    log = lambda msg: print(f"[{time.time() - started:7.0f}s] {msg}", flush=True)  # noqa: E731
    log(f"init: {sum(s.stage == 1 for s in scenes)}/{len(scenes)} scenes runnable")
    while active := [s for s in scenes if not s.done]:
        reqs = [(s.sid, *request(s, out), K) for s in active]
        responses = generator.generate(reqs)
        jobs = [(s, r) for s, rs in zip(active, responses) for r in rs]
        results = simulate_all(jobs, args)
        per = defaultdict(list)
        for (s, _), c in zip(jobs, results):
            per[s.sid].append(c)
        for s in active:
            s.generations += K
            s.simulations += len(per[s.sid])
            step(s, per[s.sid])
        stages = defaultdict(int)
        for s in scenes:
            stages["failed" if s.failed else "done" if s.done else f"stage{s.stage}"] += 1
        log(f"round done: {dict(sorted(stages.items()))}")
        save(scenes, out, init_responses)
    save(scenes, out, init_responses)
    log("finished")


def resample(args) -> None:
    """Equal-budget control, GPU part: as many fresh samples (no feedback) as the agentic run used per scene.
    Only generates (samples.jsonl); the simulations and the selection run on a CPU machine with `select`."""
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    budget = {t["scene_id"]: t["generations"] for t in read_jsonl(args.out_root / args.like / "trace.jsonl")}
    rows = {r["scene_id"]: r for r in read_jsonl(args.data / "swift" / "c4_scaffold" / "test.jsonl")}
    samples = {r["scene_id"]: r["responses"] for r in read_jsonl(args.init)}
    generator = FakeGenerator(samples) if args.fake else VllmGenerator(args.adapter, gpu_util=args.gpu_util)
    reqs = [(sid, [m for m in rows[sid]["messages"] if m["role"] != "assistant"], list(rows[sid]["images"]), n)
            for sid, n in budget.items() if n > 0]
    fresh = dict(zip([r[0] for r in reqs], generator.generate(reqs)))
    write_jsonl(out / "samples.jsonl", [{"scene_id": sid, "generations": budget[sid], "responses": fresh.get(sid, [])}
                                        for sid in budget])
    print(f"wrote {sum(len(v) for v in fresh.values())} samples to {out / 'samples.jsonl'}", flush=True)


def select(args) -> None:
    """Equal-budget control, CPU part: simulate best-of-8 plus the fresh samples; keep the runnable one that fits
    the observed frames best."""
    out = args.out_root / args.name
    fresh = {r["scene_id"]: r for r in read_jsonl(out / "samples.jsonl")}
    args.scenes = list(fresh)
    scenes = load_scenes(args)
    samples = {r["scene_id"]: r["responses"] for r in read_jsonl(args.init)}
    jobs = [(s, r) for s in scenes for r in samples[s.sid] + fresh[s.sid]["responses"]]
    results = simulate_all(jobs, args)
    per = defaultdict(list)
    for (s, _), c in zip(jobs, results):
        per[s.sid].append(c)
    rows = []
    for s in scenes:
        best = pick(per[s.sid], STAGES[-1]) or per[s.sid][0]
        rows.append({"scene_id": s.sid, "response": best.response, "generations": fresh[s.sid]["generations"],
                     "runnable": sum(c.runs for c in per[s.sid]), "candidates": len(per[s.sid])})
    write_jsonl(out / "pred.jsonl", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "resample", "select"))
    p.add_argument("--name", required=True)
    p.add_argument("--like", help="resample: the agentic run whose per-scene generation counts to match")
    p.add_argument("--data", type=Path, default=Path("data/sft_v1"))
    p.add_argument("--init", type=Path, default=Path("runs/eval/c4_scaffold__bo8/pred.jsonl"))
    p.add_argument("--adapter", help="LoRA checkpoint dir of the c4_scaffold model")
    p.add_argument("--out-root", type=Path, default=Path("runs/agentic"))
    p.add_argument("--pilot", type=int, default=30)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--sim-timeout", type=int, default=300)
    p.add_argument("--gpu-util", type=float, default=0.6)
    p.add_argument("--fake", action="store_true", help="no model: answer with best-of-8 samples (local dry run)")
    args = p.parse_args()
    {"run": run, "resample": resample, "select": select}[args.command](args)


if __name__ == "__main__":
    main()
