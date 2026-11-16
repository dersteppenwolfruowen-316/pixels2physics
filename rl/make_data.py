#!/usr/bin/env python3
"""Data for the RL runs: the Full2FPS training data (framerate/prepare_data.py --fps 2 --scaffold) plus
  eval100.json             108 test rows: every 2nd test row + the 16 frame-rate test scenes, sorted
  gt_{train,test}_samples  the ground-truth programs written as "samples" (sample = -1); their rewards set the ReST
                           threshold and calibrate the simulation time
Usage: python rl/make_data.py <SFT-2000 dir> <work dir>      (run_gpu.sh: /data/SFT-2000 $W)"""
import json
import subprocess
import sys
from pathlib import Path

this_file = Path(__file__)
this_file = this_file.resolve()
here = this_file.parent
source = Path(sys.argv[1])
work = Path(sys.argv[2])
data = work / 'fps_2s/data'


def read_jsonl(path):
    rows = []
    path = Path(path)
    text = path.read_text()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


# training data, same as Full2FPS
manifest_path = data / 'manifest.json'
if not manifest_path.exists():
    prepare = here.parent / 'framerate/prepare_data.py'
    command = [sys.executable, str(prepare), '--source', str(source), '--out', str(data), '--fps', '2', '--scaffold']
    subprocess.run(command, check=True)

# evaluation rows: every 2nd test row + the 16 frame-rate scenes
scenes_path = here.parent / 'framerate/test_scenes.json'
scenes_text = scenes_path.read_text()
old16 = json.loads(scenes_text)['test_indices']
rows = set()
for i in range(0, 200, 2):
    rows.add(i)
for i in old16:
    rows.add(i)
indices = sorted(rows)
eval_text = json.dumps({'test_indices': indices, 'old16': old16}, indent=1)
eval_path = work / 'eval100.json'
eval_path.write_text(eval_text)
count = len(indices)
print('eval rows', count)

# ground-truth programs as samples (sample = -1)
manifest_text = manifest_path.read_text()
records = json.loads(manifest_text)['records']
for split in ('train', 'test'):
    split_records = []
    for rec in records:
        if rec['split'] == split:
            split_records.append(rec)
    split_rows = read_jsonl(data / (split + '.jsonl'))
    out_path = work / ('gt_' + split + '_samples.jsonl')
    with out_path.open('w') as f:
        count = len(split_records)
        assert count == len(split_rows)
        for i in range(count):
            rec = split_records[i]
            row = split_rows[i]
            sample = {'scene_id': rec['scene_id'], 'split': split, 'row': i, 'sample': -1, 'response': row['messages'][-1]['content'], 'finish_reason': 'stop'}
            line = json.dumps(sample)
            f.write(line + '\n')
