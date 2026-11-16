# Reinforcement learning from simulation reward

Can execution feedback from the simulator improve program writing beyond supervised fine-tuning? Starting from the
Full2FPS model (`framerate/`, checkpoint 400, called **A**), we try two offline methods that need no rollout inside
the gradient update:

- **ReST-EM, one round (B1):** sample from A, keep the high-reward programs, fine-tune on them.
- **DPO on top of B1 (B1 + DPO):** preference pairs from the same samples.

## Files

| path | role |
|---|---|
| `make_data.py` | Full2FPS data (`framerate/prepare_data.py --fps 2 --scaffold`), the 108-scene eval set, ground-truth programs as samples |
| `gen_vllm.py` | fast sampling with vLLM through ms-swift (same template and video sampling as training) |
| `sim_score.py` | runs every program once and computes the `scripts/evaluate.py` metrics and the reward |
| `rest_build.py` | ReST filter: keeps samples above the per-scene threshold, writes the SFT set |
| `dpo_build.py` | builds chosen / rejected pairs |
| `train_sft.sh`, `train_dpo.sh` | ms-swift LoRA SFT (ReST M-step) and DPO |
| `analyze.py` | paired comparison of arms: greedy, best-of-k, pass@k, bootstrap CIs, McNemar |
| `run_gpu.sh` | stages `smoke genA restB1 genB1 report` (A sampling, ReST, B1 eval) |
| `run_dpo.sh` | stages `dpo genD report3` (DPO on B1, eval, 3-arm report) |
| `boot.sh`, `setup_env.sh` | RunPod setup: environments on local disk, vLLM / ms-swift patches, dataset unpacking |
| `eval100.json` | the 108 test rows used for evaluation (every 2nd test row + the 16 frame-rate scenes) |
| `A_old_prompt_tokens.jsonl` | prompt token counts of the earlier HF inference, checked by the smoke stage |
| `results/` | `analysis.{json,md}` (A vs B1 vs B1+DPO), `vsB1/`, `vsB1_ck/` (DPO step 75 vs 172), per-sample test scores, data and training stats |

## Method

**Reward.** Each program is run once:

| outcome | reward |
|---|---|
| no parseable code, or the 4,500-token limit is hit | 0 |
| parses but produces no video | 0.1 |
| runs | 0.3 + 0.35 · IoU + 0.35 · (1 − min(W-MAE, 1)), over the whole video |

The same program re-run differs by about ±0.06 (random particle initialization).

**Sampling.** A generates 4 programs per training scene (T = 1.0): 6,398 samples, 2,972 (46.5%) run, 1,229 of
1,600 scenes have at least one runnable sample.

**ReST-EM (B1).** Keep a sample if its reward ≥ τ = max(0.5, 0.7 × reward of the ground-truth program), at most 2
per scene, no duplicates: 1,700 samples from 1,020 scenes. SFT from the base model on 1,600 ground-truth programs +
1,700 kept samples, 1 epoch (413 steps), otherwise the Full2FPS settings.

**DPO (B1 + DPO).** 1,369 pairs from A's samples (off-policy). Chosen: a high-reward sample (799) or the ground
truth (570). Rejected: a program that fails (1,122) or a lower-reward runnable one (247). Mean reward gap 0.74. The
policy continues B1's LoRA and the reference is B1; sigmoid DPO, β = 0.1, + 1.0 × NLL on chosen (RPO), lr 1e-5,
1 epoch (172 steps), batch 8.

**Evaluation.** 108 test scenes, 1 greedy + 3 samples at T = 0.7, 900 s simulation limit, scored like the
frame-rate track. Differences are paired by scene with 95% parent-family bootstrap CIs; exact McNemar on greedy
success.

## Results

Sim success is greedy (best-of-4 in brackets); IoU, W-MAE and reward are greedy.

| config | fill IoU (first 70%) ↑ | sim success ↑ | IoU (held-out) ↑ | W-MAE (held-out) ↓ | reward ↑ |
|---|---|---|---|---|---|
| A (Full2FPS, step 400) | 0.108 | 26.9% (53.7%) | 0.087 | 0.905 | 0.223 |
| ReST (B1) | 0.100 | 26.9% (56.5%) | 0.084 | 0.911 | 0.221 |
| ReST + DPO | 0.031 | 12.0% (31.5%) | 0.026 | 0.980 | 0.131 |

- **ReST barely changes A.** All differences cross zero; B1 fixes 17 scenes and breaks 17 (McNemar p = 1). Kept
  samples are programs A could already write, the main failures (Taichi semantic errors, undefined names) never
  get a corrective signal, and the whole-video reward is coarse and noisy.
- **DPO makes it worse.** Greedy success 26.9% → 12.0% (p = 0.017); already worse at 0.44 epoch (14.8%). Samples
  without parseable code rise from 20 to 59 (unclosed brackets, broken indentation). The log-probability of chosen
  programs fell from about −77 to −125 nats. Easy, off-policy pairs over ~3k-token programs push the model away
  from surface text instead of teaching what makes Taichi code correct.

Next steps: repair data with the simulator's error message, on-policy pairs where both sides run, a lower lr or a
higher β, and best-of-k with an execution check at inference (already 26.9% → 53.7% for A).

## How to run

On a RunPod pod with one 96 GB GPU (RTX PRO 6000) and a network volume at `/workspace` holding `SFT-2000.zip` and
the Full2FPS checkpoint 400 at `$W/ckpt/A_ck400` (override with `A_CK=`). The whole round took about 13 GPU-hours.

```bash
bash rl/setup_env.sh          # once: train / vLLM / sim environments and the model, saved to the volume
bash rl/run_gpu.sh            # A sampling -> ReST filter -> B1 SFT -> B1 eval -> report   (resumable)
bash rl/run_dpo.sh            # DPO on B1 -> eval -> results/analysis.md                  (resumable)
```

Pinned versions: ms-swift 4.6.0, transformers 4.57.6, torch 2.8.0, vLLM 0.11.0, taichi 1.7.4. `boot.sh` patches two
bugs on every start: vLLM 0.11's Qwen3-VL LoRA module prefix (`model.visual.` → `visual.`) and ms-swift's
`shutil.copy` of `args.json`, which fails on the geesefs volume.
