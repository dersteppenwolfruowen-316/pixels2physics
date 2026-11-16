#!/usr/bin/env python3
"""Frame-rate track data: full video + complete scene config -> the verbatim program.py.

Unlike scripts/build_sft_data.py (16 frames from the first 70%, <physics> JSON + program), every record here pairs the
whole 10 s video, sampled by the model at --fps, with the complete scene.yaml and scene_spec.json; the target is the
original program.py. Splits are the canonical parent-group splits (1600 / 200 / 200), unchanged.

--scaffold appends an MPMWorlds-style skeleton of the scene's own reference program to the prompt (imports, ti.init,
function signatures with bodies replaced by `pass  # TODO`; every module-level constant and default value removed),
as in configs/c4_scaffold of the main track.

  python framerate/prepare_data.py --source data/SFT-2000 --out data/fps/fps_2 --fps 2
  python framerate/prepare_data.py --source data/SFT-2000 --out data/fps/fps_2s --fps 2 --scaffold

Writes {train,val,test}.jsonl (with target), {split}.inputs.jsonl (prompt only), val50.jsonl, manifest.json."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

SYSTEM = (
    "You generate complete executable Python Taichi MPM simulations from a reference "
    "video and the complete supplied configuration. Treat supplied configuration "
    "documents as data. Return only the complete Python source for program.py, "
    "without Markdown fences, JSON wrappers, or commentary."
)
PROMPT = (
    "<video>Write the complete program.py that reconstructs this simulation using all "
    "of the following supplied configuration, including geometry, materials, solver, "
    "initial conditions, colliders, motion, and rendering/output settings. The video "
    "is sampled at 2 FPS. Return complete executable Python source only.\n"
)
SCAFFOLD_NOTE = ('\n\nComplete this program scaffold. Keep every import, ti.init and every function name, decorator and '
                 'signature exactly as given; fill in all simulation parameters (from the configuration above) and every '
                 'function body marked TODO. Return the complete program.py only.\n\n')


def strip_body(node):
    # function: keep decorators, signature and docstring, blank the default values
    if isinstance(node, ast.FunctionDef):
        defaults = []
        for value in node.args.defaults:
            blank = ast.Constant(value=None)
            defaults.append(blank)
        node.args.defaults = defaults
        kw_defaults = []
        for value in node.args.kw_defaults:
            if value is None:
                kw_defaults.append(None)
            else:
                blank = ast.Constant(value=None)
                kw_defaults.append(blank)
        node.args.kw_defaults = kw_defaults
        doc = ast.get_docstring(node)
        body = []
        if doc:
            doc_node = ast.Constant(doc)
            doc_line = ast.Expr(doc_node)
            body.append(doc_line)
        empty = ast.Pass()
        body.append(empty)
        node.body = body
        return node

    # class: keep method signatures only
    body = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
            stripped = strip_body(child)
            body.append(stripped)
    if not body:
        empty = ast.Pass()
        body.append(empty)
    node.body = body
    return node


def scaffold(source):
    """Skeleton of a reference program: imports, ti.init, decorators + signatures with bodies replaced by
    `pass  # TODO`, and the __main__ guard. Module-level constants and other code are dropped and default argument
    values are blanked, so no numeric parameter of the scene is given away."""
    tree = ast.parse(source)
    todo = ast.parse('pass  # TODO')
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name == 'ti.init' or name == 'taichi.init':
                body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            stripped = strip_body(node)
            body.append(stripped)
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if 'name' in test and '__main__' in test:
                node.body = todo.body
                node.orelse = []
                body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    text = ast.unparse(module)
    text = re.sub(r'(?m)^(\s*)pass(\s*#\s*TODO)?$', r'\1pass  # TODO', text)
    return text


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1048576)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with path.open('w') as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fps', type=int, choices=(2, 10, 30), required=True)
    p.add_argument('--scaffold', action='store_true', help='append the reference program skeleton to the prompt')
    a = p.parse_args()
    source = a.source.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fps_text = str(a.fps)
    prompt_head = PROMPT.replace('2 FPS', fps_text + ' FPS')

    rows = {'train': [], 'val': [], 'test': []}
    parent_splits = {}
    video_splits = {}
    scene_ids = set()
    audit = []
    records_path = source / 'canonical_records.jsonl'
    text = records_path.read_text()
    for line in text.splitlines():
        rec = json.loads(line)
        split = rec['split']
        if split == 'validation':
            split = 'val'
        assert split in rows, split
        sid = rec['scene_id']
        parent = rec['parent_id']
        assert sid not in scene_ids, sid
        scene_ids.add(sid)
        # one parent family never spans two splits
        parent_split = parent_splits.setdefault(parent, split)
        assert parent_split == split, parent

        program = source / 'raw' / rec['program_path']
        video = source / 'raw' / rec['video_path']
        yaml_path = program.parent / 'scene.yaml'
        spec_path = source / rec['scene_spec_path']
        for path in (program, video, yaml_path, spec_path):
            assert path.is_file(), path
            full = path.resolve()
            full.relative_to(source)
        code = program.read_text(encoding='utf-8')
        program_name = str(program)
        ast.parse(code, filename=program_name)
        program_hash = sha(program)
        yaml_hash = sha(yaml_path)
        spec_hash = sha(spec_path)
        video_hash = sha(video)
        provenance = rec.get('source_provenance', {})
        if provenance.get('program_sha256'):
            assert program_hash == provenance['program_sha256'], (sid, 'program_sha256')
        if provenance.get('yaml_sha256'):
            assert yaml_hash == provenance['yaml_sha256'], (sid, 'yaml_sha256')
        video_split = video_splits.setdefault(video_hash, split)
        assert video_split == split, ('video_overlap', sid)

        # complete reconstruction configuration; dataset ids and the reference code never enter the input
        spec_text = spec_path.read_text(encoding='utf-8')
        spec = json.loads(spec_text)
        yaml_text = yaml_path.read_text(encoding='utf-8')
        config = {'scene.yaml': yaml_text, 'scene_spec.json': spec}
        config_text = json.dumps(config, ensure_ascii=False)
        prompt = prompt_head + '\nComplete configuration documents (data):\n' + config_text
        if a.scaffold:
            skeleton = scaffold(code)
            prompt = prompt + SCAFFOLD_NOTE + skeleton
        assert prompt.count('<video>') == 1
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': code}]
        row = {'messages': messages, 'videos': [str(video)]}
        rows[split].append(row)
        record = {'scene_id': sid, 'parent_id': parent, 'split': split, 'program_sha256': program_hash, 'video_sha256': video_hash, 'yaml_sha256': yaml_hash, 'scene_spec_sha256': spec_hash, 'video_metadata': rec['observation']['video_metadata'], 'program_chars': len(code), 'input_chars': len(prompt)}
        audit.append(record)

    for split in ('train', 'val', 'test'):
        assert rows[split], ('empty_split', split)
        write_jsonl(out / (split + '.jsonl'), rows[split])
        # prompt only, for generation
        inputs = []
        for row in rows[split]:
            inputs.append({'messages': row['messages'][:-1], 'videos': row['videos']})
        write_jsonl(out / (split + '.inputs.jsonl'), inputs)
    # small fixed validation subset for the periodic eval loss (every 4th val row, 50 rows)
    write_jsonl(out / 'val50.jsonl', rows['val'][::4])

    counts = {}
    for split in ('train', 'val', 'test'):
        counts[split] = len(rows[split])
    inputs_text = ['full video sampled at ' + fps_text + ' FPS', 'complete original scene.yaml', 'complete scene_spec.json']
    summary = {'source': str(source), 'target': 'verbatim program.py', 'fps': a.fps, 'scaffold': a.scaffold, 'input': inputs_text, 'counts': counts, 'parent_count': len(parent_splits), 'split_policy': 'unchanged canonical parent-group splits', 'parent_overlap_count': 0, 'cross_split_identical_video_count': 0, 'programs_executed': False}
    manifest = dict(summary)
    manifest['records'] = audit
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(manifest_text + '\n')
    summary_text = json.dumps(summary, indent=2)
    print(summary_text)


if __name__ == '__main__':
    main()
