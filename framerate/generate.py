#!/usr/bin/env python3
"""Generate programs for the frame-rate test scenes with a trained adapter (ms-swift, transformers backend).

Sample 0 is greedy; samples 1..N-1 use temperature 0.7 / top-p 0.95 (no fixed seed). The output has one row per
scene with `response` (greedy) and `responses` (all samples, greedy first), the format scripts/evaluate.py reads,
including --best-of-n. Resumable per (scene, sample): raw samples are appended to <out>.samples.jsonl, because a
Spot VM can be stopped in the middle of a run.

  python framerate/generate.py --data data/fps/fps_2s --fps 2 --adapter runs/framerate/fps_2s/full/checkpoint-400 \
      --samples 4 --out runs/framerate/fps_2s/full/ck400/pred.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

this_file = Path(__file__)
this_file = this_file.resolve()
HERE = this_file.parent


def read_jsonl(path):
    rows = []
    path = Path(path)
    text = path.read_text()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    with path.open('w') as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + '\n')


def merge_samples(raw_rows, indices, records):
    """One row per scene: greedy `response` and all `responses`, sample 0 first."""
    merged = []
    for i in indices:
        sid = records[i]['scene_id']
        by_sample = {}
        for row in raw_rows:
            if row['scene_id'] == sid:
                by_sample[row['sample']] = row
        responses = []
        finish_reasons = []
        completion_tokens = []
        for s in sorted(by_sample):
            responses.append(by_sample[s]['response'])
            finish_reasons.append(by_sample[s]['finish_reason'])
            completion_tokens.append(by_sample[s]['completion_tokens'])
        merged.append({'scene_id': sid, 'row': i, 'response': responses[0], 'responses': responses, 'finish_reasons': finish_reasons, 'completion_tokens': completion_tokens})
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='framerate/prepare_data.py output dir')
    p.add_argument('--fps', type=int, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--scenes', type=Path, default=HERE / 'test_scenes.json', help='test row indices to generate for')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=4500)
    a = p.parse_args()

    # vision settings are read when qwen_vl_utils is imported, so they must be set first
    os.environ['FPS'] = str(a.fps)
    os.environ['FPS_MAX_FRAMES'] = '768'
    os.environ['VIDEO_MIN_TOKEN_NUM'] = '128'
    os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'torchcodec'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCHCODEC_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    from peft import PeftModel
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template

    inputs = read_jsonl(a.data / 'test.inputs.jsonl')
    manifest_path = a.data / 'manifest.json'
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    records = []
    for rec in manifest['records']:
        if rec['split'] == 'test':
            records.append(rec)
    scenes_text = a.scenes.read_text()
    indices = json.loads(scenes_text)['test_indices']

    model, processor = get_model_processor(a.model, torch_dtype=torch.bfloat16, device_map='cuda:0', load_model=True, attn_impl='sdpa', use_hf=True)
    adapter = str(a.adapter)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    template = get_template(processor, max_length=32768)
    template.model = model
    engine = TransformersEngine(model, template=template, max_batch_size=a.batch)

    raw = a.out.with_suffix('.samples.jsonl')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if raw.exists():
        for row in read_jsonl(raw):
            done.add((row['scene_id'], row['sample']))

    for s in range(a.samples):
        if s == 0:
            config = RequestConfig(max_tokens=a.max_new_tokens, temperature=0)
        else:
            config = RequestConfig(max_tokens=a.max_new_tokens, temperature=0.7, top_p=0.95)
        todo = []
        for i in indices:
            if (records[i]['scene_id'], s) not in done:
                todo.append(i)
        todo_count = len(todo)
        for start in range(0, todo_count, a.batch):
            chunk = todo[start:start + a.batch]
            requests = []
            for i in chunk:
                request = InferRequest(messages=inputs[i]['messages'], videos=inputs[i]['videos'])
                requests.append(request)
            began = time.monotonic()
            outputs = engine.infer(requests, request_config=config)
            seconds = time.monotonic() - began
            with raw.open('a') as f:
                for i, output in zip(chunk, outputs, strict=True):
                    choice = output.choices[0]
                    response = choice.message.content
                    if response is None:
                        response = ''
                    item = {'scene_id': records[i]['scene_id'], 'row': i, 'sample': s, 'response': response, 'finish_reason': choice.finish_reason, 'completion_tokens': output.usage.completion_tokens, 'prompt_tokens': output.usage.prompt_tokens, 'batch_seconds': round(seconds, 1)}
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + '\n')
                    print(item['scene_id'], 'sample', s, item['finish_reason'], item['completion_tokens'], 'tokens')

    raw_rows = read_jsonl(raw)
    merged = merge_samples(raw_rows, indices, records)
    write_jsonl(a.out, merged)
    count = len(merged)
    print('merged', count)


if __name__ == '__main__':
    main()
