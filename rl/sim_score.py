#!/usr/bin/env python3
"""Run every generated program once and record everything needed later, one row per (scene_id, sample):
  - the evaluate.py code-track metrics, from evaluate.simulate itself (iou_observed, iou_heldout, wmae_heldout, ocs)
  - whole-video metrics for the reward: iou_all, wmae_all (same functions, all frames, start 0)
  - reward = 0 (no parsable code / hit the token cap) | 0.1 (parses, does not produce a video)
             | 0.3 + 0.35*iou_all + 0.35*(1 - min(wmae_all, 1))
--device cpu hides CUDA so simulations never compete with vLLM / training for the GPU; --device gpu keeps ti.cuda.
Resumable: rows already in --out are skipped. Videos are deleted after scoring unless --keep-videos.

Input rows (from gen_vllm.py): {scene_id, split, row, sample, response, finish_reason, ...}
Usage: sim_score.py --samples samples.jsonl --out scores.jsonl --data $W/fps_2s/data --workers 30"""
import argparse
import importlib.util
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# work dir with fps_2s/data, runs/, results/ (set by rl/boot.sh)
work_text = os.environ.get('W', '/workspace/mpm')
WORK = Path(work_text)

this_file = Path(__file__)
this_file = this_file.resolve()
here = this_file.parent
spec = importlib.util.spec_from_file_location('evaluate', here.parent / 'scripts' / 'evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def code_of(text):
    """Same code-only parsing as framerate/score.py: a fenced block if present, else the whole response."""
    physics, code = ev.parse_response(text)
    if code is None:
        code = text
    return ev.parsed_code(code)


def reward(r):
    if r['code_parses'] == 0 or r.get('finish_reason') == 'length':
        return 0.0
    if not r['sim_success']:
        return 0.1
    wmae = min(r['wmae_all'], 1.0)
    return 0.3 + 0.35 * r['iou_all'] + 0.35 * (1 - wmae)


def read_samples(path):
    # tolerant: the generator may be appending to this file right now (last line incomplete)
    samples = []
    text = path.read_text()
    for line in text.splitlines():
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples.append(sample)
    return samples


def score_sample(s, meta, timeout, vid_root, keep_root, keep_fraction, device):
    """Simulate one sample and return its score row."""
    code = code_of(s['response'])
    r = {}
    for key in ('scene_id', 'split', 'row', 'sample', 'finish_reason', 'completion_tokens'):
        r[key] = s.get(key)
    r['code_parses'] = float(code is not None)
    r.update(ev.FAILED_ROLLOUT)
    r['iou_all'] = 0.0
    r['wmae_all'] = 1.0
    r['sim_success'] = 0.0
    r['sim_seconds'] = None
    r['sim_error'] = None
    if code is None:
        r['sim_error'] = 'no_code'
    r['sim_device'] = device

    if code is not None:
        name = s['scene_id'] + '__' + str(s['sample'])
        vdir = vid_root / name
        began = time.monotonic()
        result = ev.simulate(meta, code, timeout, vdir)
        seconds = time.monotonic() - began
        r['sim_seconds'] = round(seconds, 1)
        r.update(result)
        if result['sim_success']:
            # whole-video metrics for the reward
            video = vdir / (meta['scene_id'] + '.mp4')
            true_video = Path(meta['video_path'])
            gray_true = ev.decode_gray(true_video)
            gray_pred = ev.decode_gray(video)
            fg_true = ev.foreground(gray_true)
            fg_pred = ev.foreground(gray_pred)
            truth = ev.occupancy(fg_true)
            pred = ev.occupancy(fg_pred)
            frame_count = len(truth)
            frames = range(0, frame_count, 3)
            r['iou_all'] = ev.mask_iou(truth, pred, frames)
            r['wmae_all'] = ev.windowed_motion_error(gray_true, gray_pred, 0)
            r['pred_frames'] = len(gray_pred)
            r['true_frames'] = len(gray_true)
            if random.random() < keep_fraction:
                keep_root.mkdir(parents=True, exist_ok=True)
                shutil.copy(video, keep_root / (name + '.mp4'))
        shutil.rmtree(vdir, ignore_errors=True)
    r['reward'] = reward(r)
    return r


def main():
    # default: half of the CPU cores
    cpu_count = os.cpu_count()
    workers = 1
    if cpu_count:
        workers = max(1, cpu_count // 2)
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=WORK / 'fps_2s/data')
    p.add_argument('--workers', type=int, default=workers)
    p.add_argument('--threads-per-sim', type=int, default=2)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--keep-videos', type=float, default=0.0, help='fraction of rendered videos to keep for inspection')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', choices=('cpu', 'gpu'), default='cpu', help='gpu: programs keep their ti.cuda, 1 GB Taichi pool each')
    a = p.parse_args()
    os.environ['TI_CPU_MAX_NUM_THREADS'] = str(a.threads_per_sim)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TI_LOG_LEVEL'] = 'error'
    old_path = os.environ.get('PATH', '')
    os.environ['PATH'] = '/opt/mpm/bin' + os.pathsep + old_path
    if a.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['TI_DEVICE_MEMORY_GB'] = '1'

    manifest_path = a.data / 'manifest.json'
    manifest_text = manifest_path.read_text()
    records = json.loads(manifest_text)['records']
    by_split = {'train': [], 'val': [], 'test': []}
    for rec in records:
        by_split[rec['split']].append(rec)
    split_rows = {}

    def meta_of(split, i):
        # rows of a split are read only when first needed
        if split not in split_rows:
            split_rows[split] = ev.read_jsonl(a.data / (split + '.jsonl'))
        rec = by_split[split][i]
        heldout_start = int(0.7 * rec['video_metadata']['frame_count'])
        return {'scene_id': rec['scene_id'], 'video_path': split_rows[split][i]['videos'][0], 'heldout_start_frame': heldout_start}

    samples = read_samples(a.samples)
    done = set()
    if a.out.exists():
        for row in ev.read_jsonl(a.out):
            done.add((row['scene_id'], row['sample']))
    todo = []
    for s in samples:
        if (s['scene_id'], s['sample']) not in done:
            todo.append(s)
    if a.limit:
        todo = todo[:a.limit]
    vid_root = a.out.parent / (a.out.stem + '_videos')
    keep_root = a.out.parent / (a.out.stem + '_kept')
    lock = threading.Lock()
    n_samples = len(samples)
    n_done = len(done)
    n_todo = len(todo)
    print(n_samples, 'samples,', n_done, 'done,', n_todo, 'to run,', a.workers, 'workers')

    def run(s):
        split = s.get('split', 'test')
        meta = meta_of(split, s['row'])
        return score_sample(s, meta, a.timeout, vid_root, keep_root, a.keep_videos, a.device)

    start_time = time.monotonic()
    count = 0
    with ThreadPoolExecutor(a.workers) as pool, a.out.open('a') as f:
        futures = []
        for s in todo:
            future = pool.submit(run, s)
            futures.append(future)
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as error:  # one odd sample must not stop the whole run
                message = repr(error)
                print('ERROR', message[:300])
                continue
            with lock:
                line = json.dumps(r)
                f.write(line + '\n')
                f.flush()
            count = count + 1
            if count % 25 == 0 or count == n_todo:
                elapsed = time.monotonic() - start_time
                eta = elapsed / count * (n_todo - count) / 60
                minutes = round(elapsed / 60, 1)
                eta = round(eta, 1)
                print(count, '/', n_todo, minutes, 'min, eta', eta, 'min')
    print('SIM_DONE')


if __name__ == '__main__':
    main()
