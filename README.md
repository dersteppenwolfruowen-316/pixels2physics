# pixels2physics

Pixels2Physics: Vision-to-MPM program synthesis via VLM fine-tuning and simulation-guided iterative refinement.

Project for CS 5787 Deep Learning (Fall 2026), Cornell Tech \
Ruowen Lou, Waikin Un, Chenghao Guan

---

Given the first 70% of a video of a 2D MPM simulation, a vision-language model infers the scene's physics (object
geometry, initial velocity, gravity) and writes a complete Taichi MPM program that reproduces the video. The program
is then run and its rollout compared with the held-out last 30% of the video, which the model never sees.

We study two questions:
1. **Fine-tuning.** Can LoRA SFT of a small VLM (Qwen3-VL-4B) learn to infer physics from video and to write runnable
   simulation programs?
2. **Refinement.** Can the same model improve its own program when it gets simulation feedback (error messages,
   comparison images), in a staged generator-verifier loop after SEIG / "Thinking in Blender"?

## Repository layout

| path | role |
|---|---|
| `scripts/build_sft_data.py` | builds ms-swift SFT datasets from the MPMWorlds scenes: frames, prompts per config, targets, ground-truth meta |
| `scripts/train_sft.py` | LoRA SFT with ms-swift, with a physics-weighted cross-entropy loss |
| `scripts/evaluate.py` | generation (vLLM), reference predictions (oracle, blind guess), scoring on both tracks, comparison tables |
| `scripts/run_pod.sh` | GPU part on a cloud VM: data → train → generate → upload → stop the VM |
| `scripts/score_all.sh` | CPU part: run every predicted program, score, write tables |
| `agentic/loop.py` | staged agentic refinement and the equal-budget resampling control |
| `agentic/feedback.py` | stage definitions, comparison images and text feedback, stage losses |
| `agentic/runner.py` | sandboxed Taichi rollout of a program (frame limit, traceback, static hints) |
| `agentic/run_pilot.sh` | the 30-scene refinement pilot on the GPU VM |
| `notebooks/` | dataset analysis and leakage audit (family-level overlap between scenes) |

## Data

We use 2000 scenes from the MPMWorlds dataset, each with a 300-frame video and the Taichi program that produced
it. Scenes come in families (a base scene and its variants), so we split by family: 1600 / 200 / 200 scenes for
train / validation / test.

Each training record pairs 16 frames (384 px) from the first 70% of the video (frames 1–210) with a prompt, and the
target is a `<physics>` JSON block followed by a complete Taichi program. Configs differ only in what the prompt gives:

| config | given in the prompt | the model infers |
|---|---|---|
| `c1_full_aux` | material, geometry, motion | (writes the program only) |
| `c2_no_position` | material, motion | geometry |
| `c4_no_state` | material | geometry, initial velocity, gravity |
| `c4_scaffold` | material + program scaffold (imports, field declarations, function signatures) | same as C4 |

## Method

**SFT** (`scripts/train_sft.py`). LoRA fine-tuning of Qwen3-VL-4B with [ms-swift](https://github.com/modelscope/ms-swift),
using a weighted cross-entropy that upweights the `<physics>` block and the physical values in the code.

**Agentic refinement** (`agentic/loop.py`). Starting from the best of 8 samples, the SFT model rewrites its own
program from simulation feedback in stages (executable → geometry → motion → force); a verifier keeps a rewrite only
if it fits the observed frames better. Control: equal-budget resampling without feedback.

## Evaluation

| track | metric | meaning |
|---|---|---|
| physics | geometry MAE, motion relative error, gravity log error | inferred values vs ground truth |
| physics | fill IoU (held-out) | inferred values substituted into the ground-truth program, then run |
| code | sim success | the model's own program runs and renders the video |
| code | IoU (held-out) | occupancy IoU between rollout and video on frames 211–300 |
| code | W-MAE (held-out), OCS | motion-activity error, object collapse |

References: **blind guess** (train-set mode, lower bound) and **oracle** (ground-truth program, upper bound).

## How to run

Requirements: ms-swift, vLLM, taichi 1.7.4, ffmpeg.

```bash
# 1. build the SFT data
python scripts/build_sft_data.py --data-root data/SFT-2000 --out data/sft_v1

# 2. train and generate
bash scripts/run_pod.sh c4_no_state

# 3. run the programs and score
PYTHON=.venv-eval/bin/python bash scripts/score_all.sh

# 4. agentic refinement, then score
bash agentic/run_pilot.sh
.venv-eval/bin/python scripts/evaluate.py score --pred runs/agentic/pilot/pred.jsonl --config c4_no_state --simulate
```
