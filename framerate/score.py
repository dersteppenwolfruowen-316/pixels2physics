#!/usr/bin/env python3
"""Score frame-rate predictions with scripts/evaluate.py (code track, --simulate), greedy and best-of-N.

The frame-rate data has no <physics> block, so two small adaptations are made, as in every frame-rate run:
  - a response without a ```python fence is taken as the whole program;
  - the evaluator's meta rows are built from the prepared data: ground-truth program written to disk, video path,
    held-out start = 70% of the frames.
Best-of-N takes the first sample that runs; the selection never looks at the ground truth.

  python framerate/score.py --data data/fps/fps_2s --pred runs/framerate/fps_2s/full/ck400/pred.jsonl
  # -> <pred dir>/greedy/{scores.jsonl,summary.json} and <pred dir>/best_of_4/...
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

this_file = Path(__file__)
this_file = this_file.resolve()
ROOT = this_file.parents[1]
spec = importlib.util.spec_from_file_location('evaluate', ROOT / 'scripts/evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
original_parse = ev.parse_response


def parse_code_only(text):
    # no fenced block: the whole response is the program
    physics, code = original_parse(text)
    if code is None:
        code = text
    return physics, code


ev.parse_response = parse_code_only


def build_meta(data, out):
    """evaluate.py meta rows for the frame-rate test split."""
    rows = ev.read_jsonl(data / 'test.jsonl')
    manifest_path = data / 'manifest.json'
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    test = []
    train_meta = []
    for rec in manifest['records']:
        if rec['split'] == 'test':
            test.append(rec)
        if rec['split'] == 'train':
            train_meta.append({'scene_id': rec['scene_id'], 'parent_id': rec['parent_id'], 'variant': '', 'roles': [], 'physics': [], 'gravity': {}})

    meta_dir = out / 'meta'
    program_dir = out / 'programs'
    meta_dir.mkdir(parents=True, exist_ok=True)
    program_dir.mkdir(parents=True, exist_ok=True)
    test_meta = []
    for rec, row in zip(test, rows, strict=True):
        program = program_dir / (rec['scene_id'] + '.py')
        program.write_text(row['messages'][-1]['content'])
        variant = rec['scene_id'].split('_', 2)[-1]
        heldout_start = int(0.7 * rec['video_metadata']['frame_count'])
        meta = {'scene_id': rec['scene_id'], 'parent_id': rec['parent_id'], 'variant': variant, 'roles': [], 'physics': [], 'gravity': {}, 'video_path': row['videos'][0], 'program_path': str(program), 'heldout_start_frame': heldout_start}
        test_meta.append(meta)
    ev.write_jsonl(meta_dir / 'test.jsonl', test_meta)
    ev.write_jsonl(meta_dir / 'train.jsonl', train_meta)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='framerate/prepare_data.py output dir')
    p.add_argument('--pred', type=Path, required=True, help='framerate/generate.py output')
    p.add_argument('--sim-timeout', type=int, default=900)
    p.add_argument('--sim-workers', type=int, default=4)
    p.add_argument('--greedy-only', action='store_true')
    a = p.parse_args()
    meta = a.pred.parent / 'meta_data'
    build_meta(a.data, meta)

    # number of samples per scene
    n = 1
    for row in ev.read_jsonl(a.pred):
        responses = row.get('responses', [None])
        count = len(responses)
        if count > n:
            n = count
    arms = [('greedy', False)]
    if not a.greedy_only and n > 1:
        n_text = str(n)
        arms.append(('best_of_' + n_text, True))

    for arm, best_of_n in arms:
        out = a.pred.parent / arm
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.pred, out / 'pred.jsonl')
        print('== scoring', arm)
        args = SimpleNamespace(data=meta, split='test', pred=out / 'pred.jsonl', out_dir=out, config='c1_full_aux', simulate=True, fill=False, best_of_n=best_of_n, sim_timeout=a.sim_timeout, sim_workers=a.sim_workers, bootstrap=1000, wandb_project=None, wandb_name=None)
        ev.score(args)


if __name__ == '__main__':
    main()
