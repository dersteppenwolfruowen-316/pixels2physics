#!/usr/bin/env python3
"""ReST-EM M-step data: original 1,600 ground-truth rows  U  filtered self-generated programs.
A sample is kept if it ran and reward >= tau(scene) = max(--tau-floor, --tau-frac * reward of the scene's own
ground-truth program). At most --per-scene samples per scene, highest reward first, exact duplicates dropped.
The kept target is the parsed program (same code-only parsing as scoring), the prompt is the scene's training prompt.

Usage: rest_build.py --samples train_samples.jsonl --scores train_scores.jsonl --gt gt_scores.jsonl --out rest_train.jsonl"""
import argparse
import copy
import json
import os
import random
from pathlib import Path

from sim_score import code_of

# work dir with fps_2s/data, runs/, results/ (set by rl/boot.sh)
work_text = os.environ.get('W', '/workspace/mpm')
WORK = Path(work_text)


def read_jsonl(path):
    rows = []
    path = Path(path)
    text = path.read_text()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


def reward_of(r):
    return r['reward']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=WORK / 'fps_2s/data')
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--gt', type=Path, help='scores of the ground-truth programs (sample = -1)')
    p.add_argument('--tau-frac', type=float, default=0.7)
    p.add_argument('--tau-floor', type=float, default=0.5)
    p.add_argument('--per-scene', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = read_jsonl(a.data / 'train.jsonl')
    responses = {}
    for s in read_jsonl(a.samples):
        responses[(s['scene_id'], s['sample'])] = s['response']
    gt = {}
    if a.gt:
        for r in read_jsonl(a.gt):
            gt[r['scene_id']] = r['reward']
    by_scene = {}
    for r in read_jsonl(a.scores):
        if r['scene_id'] not in by_scene:
            by_scene[r['scene_id']] = []
        by_scene[r['scene_id']].append(r)

    kept = []
    stats = {'scenes': 0, 'scenes_any_run': 0, 'scenes_kept': 0, 'dupes': 0, 'samples': 0, 'samples_run': 0}
    for sid in by_scene:
        scores = by_scene[sid]
        stats['scenes'] = stats['scenes'] + 1
        for r in scores:
            if r['sim_success']:
                stats['scenes_any_run'] = stats['scenes_any_run'] + 1
                break
        gt_reward = gt.get(sid, 0.0)
        tau = max(a.tau_floor, a.tau_frac * gt_reward)
        seen = set()
        n = 0
        # highest reward first (stable for ties)
        ranked = sorted(scores, key=reward_of, reverse=True)
        for r in ranked:
            if not r['sim_success'] or r['reward'] < tau or n >= a.per_scene:
                continue
            response = responses[(sid, r['sample'])]
            code = code_of(response)
            if code is None:
                continue
            if code in seen:
                stats['dupes'] = stats['dupes'] + 1
                continue
            seen.add(code)
            n = n + 1
            row = copy.deepcopy(rows[r['row']])
            row['messages'][-1]['content'] = code
            kept.append(row)
        if n > 0:
            stats['scenes_kept'] = stats['scenes_kept'] + 1
        stats['samples'] = stats['samples'] + len(scores)
        for r in scores:
            stats['samples_run'] = stats['samples_run'] + r['sim_success']

    stats['kept'] = len(kept)
    out = rows + kept
    rng = random.Random(42)
    rng.shuffle(out)
    with a.out.open('w') as f:
        for row in out:
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + '\n')
    stats['train_rows'] = len(out)
    stats_text = json.dumps(stats, indent=1)
    print(stats_text)
    stats_path = a.out.with_suffix('.stats.json')
    stats_path.write_text(stats_text)


if __name__ == '__main__':
    main()
