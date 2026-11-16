# MPMWorlds closed-loop physics inference with Qwen3-VL

The workflow fits a 2D material point method scene to eight observed SFT-2000 frames. Qwen proposes scene parameters, while a fixed CPU Taichi MLS-MPM solver compiles and runs each valid candidate. The learner uses the observed prefix to revise its proposal and select a candidate before held-out evaluation.

## Observe → Infer → Verify → Revise

| Stage | Input and operation | Record |
| --- | --- | --- |
| Observe | Decode canonical prefix frames. Measure foreground coverage and centroid for each frame, coarse color boxes on the first frame, and the background color. Read the canonical time step and optional scene caption. | `observation.json` |
| Infer | Send the observations and all eight frames to Qwen3-VL-4B-Instruct. Ask for a concise physical hypothesis, an executable scene configuration, and an optional prediction of last-prefix-frame foreground coverage. | `round_XX/proposal.json`, `scene.json`, `model.txt` |
| Verify | Validate the scene and detect duplicates. Render aligned prefix frames and measure agreement, the worst-overlap frame, and signed foreground area bias. Compare the model's numeric prediction with both rendered and observed foreground coverage. | `round_XX/verification.json`, `prefix_metrics.json`, `prefix_comparison.jpg` |
| Revise | Give Qwen the latest verifier report, the tested scene when relevant, and the strongest prior candidate with its prefix comparison. A schema error, repeated scene, or poor fit provides evidence for the next proposal. Verify the revision before deciding whether to accept it. | The next `round_XX` and the acceptance decisions in `trace.json` |

The first verified scene becomes the incumbent. A revision replaces it if prefix loss improves by more than `1e-5`. The loop stops at the configured round budget or optional `--target-loss`. It writes `selection.json` before decoding held-out reference frames. If every proposal fails, the saved verifier reports document the failures and the run ends without evaluating the suffix.

## Run in Colab

1. Open `MPMWorlds_Qwen3VL_Colab.ipynb` and select a GPU runtime.
2. Run the configuration and dependency cells. The default backend loads `Qwen/Qwen3-VL-4B-Instruct` with Transformers. Set `USE_4BIT = True` to enable 4-bit loading if needed.
3. Upload `MPMWorlds_Colab_Source.zip` when prompted. The bundle contains the starter test case. To use the full dataset, set `MOUNT_DRIVE = True`, enter the ZIP or extracted directory in `FULL_DATA`, and set `SCENE_ID` and `SPLIT`.
4. Run the loop, inspect the hypothesis and verification results, and download the result ZIP from Colab's Files panel.

For OpenAI-compatible Chat Completions, set `BACKEND = "api"`, `API_BASE_URL`, and `MODEL_ID`. Set the `QWEN_API_KEY` Colab secret if the service requires authentication. The client sends observed frames and available prefix comparison images to the endpoint you configure. [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [Multimodal API reference](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html)

## Run from the command line

```bash
python -m pip install taichi==1.7.4 "Pillow>=10,<13"
# Install FFmpeg or set MPM_FFMPEG to the path to its executable.
python agent.py --data /path/to/SFT-2000.zip --scene scene_000234_base \
  --backend api --base-url http://127.0.0.1:8097/v1 --json-mode \
  --split test --rounds 3 --description --out runs/scene_000234_base
```

Use `--json-mode` for endpoints that support JSON object output. `--data` also accepts an extracted directory with `canonical_records.jsonl`. The ZIP reader stages only the selected reference video locally. `--description` exposes the dataset caption, as in the bundled Colab example. The dataset's prompts, Python solvers, and training targets are source data. Executable candidates consist of validated JSON and the bundled solver.

The validator normalizes integer RGB triplets to hex colors and interprets a scalar gravity value as the magnitude of downward acceleration. Two-component gravity vectors preserve the proposed direction.

## Evidence and metrics

- The starter scene uses zero-based frames **13, 39, 65, 91, 118, 144, 170, and 196** for observations and **210–299** for held-out scoring. Canonical physical timing, or the recorded fallback, controls the simulation. The video FPS controls playback.
- Prefix loss is `0.65 × (1 − mean foreground IoU) + 0.25 × foreground RGB MAE + 0.10 × motion MAE`. Both MAEs divide by 255. Motion MAE compares absolute RGB differences between consecutive samples. A pixel belongs to the foreground when its maximum channel difference from the background exceeds 28.
- The late-foreground prediction is an explicit check of agreement between the hypothesis and the simulated prefix, whose reference coverage is already available. Held-out metrics assess extrapolation after candidate selection. `summary.json` records both scores and the selected hypothesis.
- The solver supports box and circle particle bodies, static segment and circle obstacles, and approximate elastic, water, snow, and jelly materials. Driven colliders, emitters, FEM, and more complex materials fall outside this solver's scope. The bundled metrics are single-scene diagnostics using a simplified renderer. [MPMWorlds evaluation protocol](https://arxiv.org/html/2606.01538v1#S4)
