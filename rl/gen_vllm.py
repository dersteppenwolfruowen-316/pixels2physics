#!/usr/bin/env python3
"""Fast generation with ms-swift's VllmEngine (same template and qwen_vl_utils video sampling as training).
One row per (scene, sample) in --out, resumable per scene. Sample 0 is greedy when --greedy-first, the rest use
--temperature / --top-p (n samples per request, so each video is encoded once).

  gen_vllm.py --split test --indices $W/eval100.json --adapter <ckpt> --samples 4 --greedy-first \
              --temperature 0.7 --out runs/A/test_samples.jsonl          # evaluation
  gen_vllm.py --split train --adapter <ckpt> --samples 4 --temperature 1.0 --out runs/A/train_samples.jsonl  # ReST
"""
import argparse
import json
import os
import time
from pathlib import Path

# work dir with fps_2s/data, runs/, results/ (set by rl/boot.sh)
work_text = os.environ.get('W', '/workspace/mpm')
WORK = Path(work_text)

# ms-swift 4.6 ships a vLLM plugin (UEmbed) that imports a module missing in vLLM 0.11; it is not needed
os.environ.setdefault('VLLM_PLUGINS', 'none')


def read_jsonl(path):
    rows = []
    path = Path(path)
    text = path.read_text()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=WORK / 'fps_2s/data')
    p.add_argument('--split', default='test')
    p.add_argument('--indices', type=Path, help='json with test_indices (default: every row of the split)')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--adapter', type=Path, help='LoRA checkpoint (omit for a merged model given by --model)')
    p.add_argument('--model', default=None)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--greedy-first', action='store_true')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=4500)
    p.add_argument('--chunk', type=int, default=64, help='scenes per engine.infer call (results saved per chunk)')
    p.add_argument('--gpu-mem', type=float, default=0.85)
    p.add_argument('--eager', action='store_true', help='skip torch.compile + CUDA graphs (capture can stall if the CPUs are busy)')
    p.add_argument('--max-num-seqs', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    # vision settings are read when qwen_vl_utils is imported, so they must be set first
    os.environ['FPS'] = '2'
    os.environ['FPS_MAX_FRAMES'] = '768'
    os.environ['VIDEO_MIN_TOKEN_NUM'] = '128'
    os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'torchcodec'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCHCODEC_NUM_THREADS'] = '2'
    from swift.infer_engine import InferRequest, RequestConfig, VllmEngine

    inputs = read_jsonl(a.data / (a.split + '.inputs.jsonl'))
    manifest_path = a.data / 'manifest.json'
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    records = []
    for rec in manifest['records']:
        if rec['split'] == a.split:
            records.append(rec)
    assert len(inputs) == len(records)
    if a.indices:
        indices_text = a.indices.read_text()
        indices = json.loads(indices_text)['test_indices']
    else:
        count = len(records)
        indices = []
        for i in range(count):
            indices.append(i)
    if a.limit:
        indices = indices[:a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for row in read_jsonl(a.out):
            done.add(row['scene_id'])
    todo = []
    for i in indices:
        if records[i]['scene_id'] not in done:
            todo.append(i)
    n_indices = len(indices)
    n_done = len(done)
    n_todo = len(todo)
    print(n_indices, 'scenes,', n_done, 'done,', n_todo, 'to generate')
    if not todo:
        return

    model = a.model
    if model is None:
        model_path = a.data.parent.parent / 'model_path.txt'
        model_text = model_path.read_text()
        model = model_text.strip()
    adapters = None
    if a.adapter:
        adapters = [str(a.adapter)]
    engine = VllmEngine(model, adapters=adapters, max_lora_rank=64, gpu_memory_utilization=a.gpu_mem, max_model_len=12288, max_num_seqs=a.max_num_seqs, limit_mm_per_prompt={'video': 1, 'image': 0}, seed=a.seed, use_hf=True, enable_prefix_caching=True, enforce_eager=a.eager)
    greedy_config = RequestConfig(max_tokens=a.max_tokens, temperature=0, return_details=True)
    n_sampled = a.samples
    if a.greedy_first:
        n_sampled = a.samples - 1
    sampled_config = None
    if n_sampled > 0:
        sampled_config = RequestConfig(max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p, n=n_sampled, seed=a.seed, return_details=True)

    start_time = time.monotonic()
    tokens = 0
    with a.out.open('a') as f:
        for start in range(0, n_todo, a.chunk):
            chunk = todo[start:start + a.chunk]
            requests = []
            for i in chunk:
                request = InferRequest(messages=inputs[i]['messages'], videos=inputs[i]['videos'])
                requests.append(request)
            chunk_start = time.monotonic()
            greedy_out = None
            sampled_out = None
            if a.greedy_first:
                greedy_out = engine.infer(requests, request_config=greedy_config, use_tqdm=False)
            if sampled_config:
                sampled_out = engine.infer(requests, request_config=sampled_config, use_tqdm=False)

            for j, i in enumerate(chunk):
                # (response, choice) pairs: greedy first, then the sampled ones
                choices = []
                if greedy_out is not None:
                    choices.append((greedy_out[j], greedy_out[j].choices[0]))
                    tokens = tokens + greedy_out[j].usage.completion_tokens
                if sampled_out is not None:
                    for choice in sampled_out[j].choices:
                        choices.append((sampled_out[j], choice))
                    tokens = tokens + sampled_out[j].usage.completion_tokens
                for s, (response, choice) in enumerate(choices):
                    token_ids = getattr(choice, 'token_ids', None)
                    n_tokens = None
                    if token_ids:
                        n_tokens = len(token_ids)
                    text = choice.message.content
                    if text is None:
                        text = ''
                    is_greedy = a.greedy_first and s == 0
                    item = {'scene_id': records[i]['scene_id'], 'split': a.split, 'row': i, 'sample': s, 'greedy': is_greedy, 'response': text, 'finish_reason': choice.finish_reason, 'completion_tokens': n_tokens, 'prompt_tokens': response.usage.prompt_tokens}
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + '\n')
            f.flush()

            finished = start + len(chunk)
            elapsed = time.monotonic() - start_time
            chunk_seconds = time.monotonic() - chunk_start
            rate = tokens / elapsed
            eta = elapsed / finished * (n_todo - finished) / 60
            chunk_seconds = round(chunk_seconds)
            rate = round(rate)
            eta = round(eta, 1)
            print(finished, '/', n_todo, 'scenes, chunk', chunk_seconds, 's,', rate, 'tok/s, eta', eta, 'min')
    print('GEN_DONE')


if __name__ == '__main__':
    main()
