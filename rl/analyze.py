#!/usr/bin/env python3
"""Compare arms on the evaluation scenes from sim_score.py outputs (one row per scene x sample, sample 0 greedy).
Per arm: greedy and best-of-k (evaluate.py semantics: first sample that runs), pass@k (unbiased, n samples),
quality conditional on running. Paired arm-minus-baseline differences with a parent-family cluster bootstrap 95% CI
and an exact McNemar test on greedy runnability. Everything twice: all eval scenes and the old 16.

Usage: analyze.py --arm A=runs/A/test_scores.jsonl --arm B1=runs/B1/test_scores.jsonl --baseline A --out results/"""
import argparse
import json
import math
import os
import random
from pathlib import Path

# work dir with fps_2s/data, runs/, results/ (set by rl/boot.sh)
work_text = os.environ.get('W', '/workspace/mpm')
WORK = Path(work_text)

METRICS = ('code_parses', 'sim_success', 'iou_observed', 'iou_heldout', 'wmae_heldout', 'ocs', 'reward')


def read_jsonl(path):
    rows = []
    path = Path(path)
    text = path.read_text()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


def mean(values):
    """Mean of the values that are not None; None if there are none."""
    kept = []
    for x in values:
        if x is not None:
            kept.append(x)
    if not kept:
        return None
    # built-in sum: compensated float summation
    total = sum(kept)
    return total / len(kept)


def per_scene(rows, k):
    """scene -> dict of greedy / best-of-k / pass@k values."""
    by_scene = {}
    for r in rows:
        if r['scene_id'] not in by_scene:
            by_scene[r['scene_id']] = {}
        by_scene[r['scene_id']][r['sample']] = r

    out = {}
    for sid in by_scene:
        sample_rows = by_scene[sid]
        samples = []
        for i in sorted(sample_rows):
            if i < k:
                samples.append(sample_rows[i])
        greedy = sample_rows[0]
        # best-of-k: first sample that runs, else the first sample
        first_ok = samples[0]
        for x in samples:
            if x['sim_success']:
                first_ok = x
                break
        n = len(samples)
        oks = []
        for x in samples:
            if x['sim_success']:
                oks.append(x)
        c = len(oks)

        d = {}
        for m in METRICS:
            d['greedy_' + m] = greedy[m]
        for m in METRICS:
            d['bo' + str(k) + '_' + m] = first_ok[m]
        for j in range(1, n + 1):
            # unbiased pass@j from n samples with c successes
            if n - c >= j:
                fail = math.comb(n - c, j) / math.comb(n, j)
            else:
                fail = 0.0
            d['pass@' + str(j)] = 1 - fail
        d['greedy_iou_heldout_ok'] = None
        d['greedy_wmae_heldout_ok'] = None
        if greedy['sim_success']:
            d['greedy_iou_heldout_ok'] = greedy['iou_heldout']
            d['greedy_wmae_heldout_ok'] = greedy['wmae_heldout']
        d['any_iou_heldout_ok'] = None
        d['any_wmae_heldout_ok'] = None
        if oks:
            ious = []
            wmaes = []
            for x in oks:
                ious.append(x['iou_heldout'])
                wmaes.append(x['wmae_heldout'])
            d['any_iou_heldout_ok'] = mean(ious)
            d['any_wmae_heldout_ok'] = mean(wmaes)
        capped = 0
        tokens = 0
        for x in samples:
            if x.get('finish_reason') == 'length':
                capped = capped + 1
            if x.get('completion_tokens'):
                tokens = tokens + x['completion_tokens']
        d['hit_length_cap'] = capped / n
        d['completion_tokens'] = tokens / n
        out[sid] = d
    return out


def cluster_boot(diffs, families, rounds=2000, seed=0):
    """95% CI of the mean paired difference, resampling parent families (scenes of one parent move together)."""
    by_family = {}
    for d, f in zip(diffs, families):
        if d is not None:
            if f not in by_family:
                by_family[f] = []
            by_family[f].append(d)
    keys = list(by_family)
    if not keys:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(rounds):
        # draw the families first, then collect their scenes (same random order as the original version)
        picked_families = []
        for _ in keys:
            family = rng.choice(keys)
            picked_families.append(family)
        pick = []
        for family in picked_families:
            for x in by_family[family]:
                pick.append(x)
        total = sum(pick)
        count = len(pick)
        stats.append(total / count)
    stats.sort()
    low = stats[int(0.025 * rounds)]
    high = stats[int(0.975 * rounds) - 1]
    return [low, high]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired binary outcomes: b01 = base fails & arm runs, b10 = base runs & arm fails."""
    b01 = 0
    b10 = 0
    for x, y in zip(a, b):
        if not x and y:
            b01 = b01 + 1
        if x and not y:
            b10 = b10 + 1
    n = b01 + b10
    if n == 0:
        return {'arm_only': 0, 'base_only': 0, 'p': 1.0}
    k = min(b01, b10)
    tail = 0
    for i in range(k + 1):
        tail = tail + math.comb(n, i)
    p = min(1.0, 2 * tail / 2 ** n)
    return {'arm_only': b01, 'base_only': b10, 'p': p}


def table_lines(result, k):
    """Markdown tables of the main columns, one per subset."""
    k_text = str(k)
    cols = ['greedy_sim_success', 'bo' + k_text + '_sim_success', 'pass@1', 'pass@' + k_text, 'greedy_iou_heldout', 'greedy_wmae_heldout', 'greedy_iou_heldout_ok', 'greedy_wmae_heldout_ok', 'greedy_reward']
    lines = []
    for subset in result['subsets']:
        sub = result['subsets'][subset]
        n_text = str(sub['n'])
        lines.append('\n## ' + subset + ' (n=' + n_text + ')\n')
        header = ' | '.join(cols)
        lines.append('| arm | ' + header + ' |')
        col_count = len(cols)
        lines.append('|---' * (col_count + 1) + '|')
        for name in sub['arms']:
            values = sub['arms'][name]
            cells = []
            for c in cols:
                if values[c] is None:
                    cells.append('–')
                else:
                    cells.append(f'{values[c]:.3f}')
            row_text = ' | '.join(cells)
            lines.append('| ' + name + ' | ' + row_text + ' |')
        for name in sub['diffs']:
            diff = sub['diffs'][name]
            cells = []
            for c in cols:
                if diff[c]['mean'] is None:
                    cells.append('–')
                else:
                    low = diff[c]['ci95'][0]
                    high = diff[c]['ci95'][1]
                    cells.append(f"{diff[c]['mean']:+.3f} [{low:+.3f}, {high:+.3f}]")
            row_text = ' | '.join(cells)
            lines.append('| ' + name + ' | ' + row_text + ' |')
            m = diff['mcnemar_greedy_run']
            lines.append(f"\nMcNemar greedy runs {name}: arm-only {m['arm_only']}, base-only {m['base_only']}, p={m['p']:.3g}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', action='append', required=True, help='NAME=scores.jsonl')
    p.add_argument('--baseline', default='A')
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--data', type=Path, default=WORK / 'fps_2s/data')
    p.add_argument('--eval', type=Path, default=WORK / 'eval100.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    k = a.k

    manifest_path = a.data / 'manifest.json'
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    records = {}
    test = []
    for rec in manifest['records']:
        records[rec['scene_id']] = rec
        if rec['split'] == 'test':
            test.append(rec)
    eval_text = a.eval.read_text()
    old16 = set()
    for i in json.loads(eval_text)['old16']:
        old16.add(test[i]['scene_id'])

    arms = {}
    for spec in a.arm:
        name, path = spec.split('=', 1)
        rows = []
        for r in read_jsonl(path):
            if r['sample'] >= 0:
                rows.append(r)
        arms[name] = per_scene(rows, k)

    # scenes scored in every arm
    common = None
    for name in arms:
        scenes = set(arms[name])
        if common is None:
            common = scenes
        else:
            common = common & scenes
    common = sorted(common)

    k_text = str(k)
    bo_key = 'bo' + k_text + '_sim_success'
    keys = []
    for m in METRICS:
        keys.append('greedy_' + m)
    for m in METRICS:
        keys.append('bo' + k_text + '_' + m)
    for j in range(1, k + 1):
        j_text = str(j)
        keys.append('pass@' + j_text)
    keys = keys + ['greedy_iou_heldout_ok', 'greedy_wmae_heldout_ok', 'any_iou_heldout_ok', 'any_wmae_heldout_ok', 'hit_length_cap', 'completion_tokens']

    old16_scenes = []
    for s in common:
        if s in old16:
            old16_scenes.append(s)
    result = {'n_scenes': len(common), 'k': k, 'subsets': {}}
    base = arms.get(a.baseline)
    for subset, scenes in (('all', common), ('old16', old16_scenes)):
        families = []
        for s in scenes:
            families.append(records[s]['parent_id'])
        sub = {'n': len(scenes), 'arms': {}, 'diffs': {}}
        for name in arms:
            d = arms[name]
            summary = {}
            for key in keys:
                values = []
                for s in scenes:
                    value = d[s].get(key)
                    values.append(value)
                summary[key] = mean(values)
            n_ok = 0
            for s in scenes:
                n_ok = n_ok + d[s]['greedy_sim_success']
            summary['n_ok_greedy'] = n_ok
            sub['arms'][name] = summary

        for name in arms:
            if name == a.baseline or base is None:
                continue
            d = arms[name]
            diff = {}
            for key in keys:
                diffs = []
                for s in scenes:
                    x = d[s].get(key)
                    y = base[s].get(key)
                    if x is not None and y is not None:
                        diffs.append(x - y)
                    else:
                        diffs.append(None)
                diff[key] = {'mean': mean(diffs), 'ci95': cluster_boot(diffs, families)}
            base_greedy = []
            arm_greedy = []
            base_bo = []
            arm_bo = []
            for s in scenes:
                base_greedy.append(base[s]['greedy_sim_success'])
                arm_greedy.append(d[s]['greedy_sim_success'])
                base_bo.append(base[s][bo_key])
                arm_bo.append(d[s][bo_key])
            diff['mcnemar_greedy_run'] = mcnemar(base_greedy, arm_greedy)
            diff['mcnemar_bo_run'] = mcnemar(base_bo, arm_bo)
            sub['diffs'][name + '-' + a.baseline] = diff
        result['subsets'][subset] = sub

    # compute-matched: baseline best-of-k vs each arm's greedy
    result['compute_matched'] = {}
    for name in arms:
        if name == a.baseline or a.baseline not in arms:
            continue
        arm_greedy = []
        base_bo = []
        for s in common:
            arm_greedy.append(arms[name][s]['greedy_sim_success'])
            base_bo.append(arms[a.baseline][s][bo_key])
        result['compute_matched'][name] = {'arm_greedy_run': mean(arm_greedy), 'base_bo' + k_text + '_run': mean(base_bo)}

    result['per_scene'] = {}
    for name in arms:
        result['per_scene'][name] = {}
        for s in common:
            result['per_scene'][name][s] = arms[name][s]

    a.out.mkdir(parents=True, exist_ok=True)
    result_text = json.dumps(result, indent=1)
    result_path = a.out / 'analysis.json'
    result_path.write_text(result_text)
    lines = table_lines(result, k)
    table = '\n'.join(lines)
    table_path = a.out / 'analysis.md'
    table_path.write_text(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
