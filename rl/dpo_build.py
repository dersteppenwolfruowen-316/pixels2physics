#!/usr/bin/env python3
"""DPO pairs for B1 from A's already-scored train samples (no new generation; off-policy for B1, noted in the report).
Per scene, at most --per-scene pairs:
  chosen   = the scene's best sample if it ran and reward >= tau(scene) (same tau as the ReST filter), else the
             ground-truth program
  rejected = another sample whose reward is at least --gap below the chosen one's (failed runs count: 0.1, or 0 if no
             code); samples that hit the length cap or have no parsable code are skipped, as are exact duplicates
Both sides are the parsed program only, the same code-only target B1 was trained on.
Output rows: ms-swift DPO format {messages: [..., assistant=chosen], rejected_response, videos}.

Usage: dpo_build.py --samples train_samples.jsonl --scores train_scores.jsonl --gt gt_train_scores.jsonl --out dpo.jsonl"""
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


def reward_of(candidate):
    return candidate['reward']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=WORK / 'fps_2s/data')
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--tau-frac', type=float, default=0.7)
    p.add_argument('--tau-floor', type=float, default=0.5)
    p.add_argument('--gap', type=float, default=0.3)
    p.add_argument('--gt-reward', type=float, default=0.9, help='assumed GT reward where it was not simulated')
    p.add_argument('--per-scene', type=int, default=1)
    p.add_argument('--max-pairs', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = read_jsonl(a.data / 'train.jsonl')
    responses = {}
    for s in read_jsonl(a.samples):
        responses[(s['scene_id'], s['sample'])] = s['response']
    gt = {}
    for r in read_jsonl(a.gt):
        gt[r['scene_id']] = r['reward']
    by_scene = {}
    for r in read_jsonl(a.scores):
        if r['scene_id'] not in by_scene:
            by_scene[r['scene_id']] = []
        by_scene[r['scene_id']].append(r)

    rng = random.Random(42)
    out = []
    stats = {}
    gap_sum = 0.0
    chosen_count = 0
    for sid in by_scene:
        scores = by_scene[sid]
        row_i = scores[0]['row']
        gt_code = rows[row_i]['messages'][-1]['content']
        gt_reward = gt.get(sid, 0.0)
        tau = max(a.tau_floor, a.tau_frac * gt_reward)

        # candidates: samples with parsable code that did not hit the length cap
        candidates = []
        for r in scores:
            if r.get('finish_reason') == 'length':
                continue
            response = responses[(sid, r['sample'])]
            code = code_of(response)
            if code is None:
                continue
            candidates.append({'reward': r['reward'], 'ran': r['sim_success'], 'code': code})
        candidates.sort(key=reward_of, reverse=True)

        # chosen: the best sample if it is good enough, else the ground truth
        if candidates and candidates[0]['ran'] and candidates[0]['reward'] >= tau:
            chosen_reward = candidates[0]['reward']
            chosen_code = candidates[0]['code']
            source = 'sample'
        else:
            chosen_reward = gt.get(sid, a.gt_reward)
            chosen_code = gt_code
            source = 'gt'

        rejected = []
        for c in candidates:
            if c['reward'] <= chosen_reward - a.gap and c['code'] != chosen_code:
                rejected.append(c)
        if not rejected:
            stats['no_rejected'] = stats.get('no_rejected', 0) + 1
            continue
        rng.shuffle(rejected)
        for c in rejected[:a.per_scene]:
            row = copy.deepcopy(rows[row_i])
            row['messages'][-1]['content'] = chosen_code
            row['rejected_response'] = c['code']
            out.append(row)
            key = 'chosen_' + source
            stats[key] = stats.get(key, 0) + 1
            if c['ran']:
                key = 'rejected_low_reward'
            else:
                key = 'rejected_failed'
            stats[key] = stats.get(key, 0) + 1
            gap_sum = gap_sum + chosen_reward - c['reward']
            chosen_count = chosen_count + 1

    rng.shuffle(out)
    if a.max_pairs:
        out = out[:a.max_pairs]
    stats['pairs'] = len(out)
    stats['scenes'] = len(by_scene)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w') as f:
        for row in out:
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + '\n')
    mean_gap = gap_sum / max(1, chosen_count)
    stats['mean_gap'] = round(mean_gap, 3)
    stats_text = json.dumps(stats, indent=1)
    stats_path = a.out.parent / (a.out.stem + '_stats.json')
    stats_path.write_text(stats_text)
    print(stats_text)


if __name__ == '__main__':
    main()
