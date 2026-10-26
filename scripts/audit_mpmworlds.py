#!/usr/bin/env python3
"""Read-only MPMWorlds data inventory. Does not execute untrusted simulation programs."""
import argparse, ast, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
import yaml

NAME_RE = re.compile(r'^scene_(\d+)_(base|modification|perturbation)(?:_(\d+))?$')
META = ('scene_id', 'parent_id', 'variant', 'variant_id', 'part', 'relative_path', 'has_yaml', 'has_program', 'has_video', 'yaml_error', 'program_syntax_ok', 'program_lines', 'program_bytes', 'program_sha256', 'video_bytes', 'object_count')

def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def flatten(prefix, value, output):
    if isinstance(value, dict):
        for k, v in value.items(): flatten(f'{prefix}.{k}' if prefix else str(k), v, output)
    elif isinstance(value, list):
        if all(not isinstance(v, (list, dict)) for v in value):
            output[prefix] = value
        else:
            for i, v in enumerate(value): flatten(f'{prefix}[{i}]', v, output)
    else:
        output[prefix] = value

def part_name(scene_dir):
    return next((p.name for p in scene_dir.parents if p.name.startswith('mpm_worlds_data_part_')), '')

def discover(root):
    # Includes a scene with a missing YAML; avoids silently dropping broken records.
    return sorted(p for p in root.rglob('scene_*') if p.is_dir() and NAME_RE.fullmatch(p.name) and p.parent.name == 'instances')

def scan(root):
    rows, parameter_rows, object_rows, flat_by_scene = [], [], [], {}
    for scene in discover(root):
        parent, variant, variant_id = NAME_RE.fullmatch(scene.name).groups()
        yfile, pfile, vfile = (scene / x for x in ('scene.yaml', 'program.py', 'video.mp4'))
        row = dict(scene_id=scene.name, parent_id=parent, variant=variant, variant_id=variant_id or '', part=part_name(scene), relative_path=str(scene.relative_to(root)), has_yaml=yfile.is_file(), has_program=pfile.is_file(), has_video=vfile.is_file(), yaml_error='', program_syntax_ok='', program_lines='', program_bytes='', program_sha256='', video_bytes=vfile.stat().st_size if vfile.is_file() else '', object_count='')
        config = None
        if yfile.is_file():
            try:
                config = yaml.safe_load(yfile.read_text(encoding='utf-8'))
                if not isinstance(config, dict): raise ValueError('YAML root is not a mapping')
            except Exception as e: row['yaml_error'] = str(e)
        if pfile.is_file():
            raw = pfile.read_bytes()
            row.update(program_bytes=len(raw), program_lines=raw.count(b'\n')+1, program_sha256=hashlib.sha256(raw).hexdigest())
            try: ast.parse(raw.decode('utf-8')); row['program_syntax_ok'] = True
            except Exception: row['program_syntax_ok'] = False
        if config is not None:
            objs = config.get('objects', [])
            if not isinstance(objs, list):
                row['yaml_error'] = 'objects field is not a list'; objs = []
            row['object_count'] = len(objs)
            # Key full flattened paths by object index (for comparison) and fields by semantic class (for distributions).
            full = {}
            for section, value in config.items():
                if section != 'objects': flatten(section, value, full)
            for idx, obj in enumerate(objs):
                if not isinstance(obj, dict): continue
                name = str(obj.get('name', f'obj_{idx}'))
                typ = str(obj.get('object_type', 'unknown'))
                material = str(obj.get('physics_properties', {}).get('constitutive_model', 'unknown')) if isinstance(obj.get('physics_properties'), dict) else 'unknown'
                object_rows.append(dict(scene_id=scene.name,parent_id=parent,variant=variant,object_name=name,object_type=typ,material=material))
                flat = {}; flatten('', obj, flat)
                for key, val in flat.items():
                    full[f'objects.{name}.{key}'] = val
                    parameter_rows.append(dict(scene_id=scene.name,parent_id=parent,variant=variant,object_name=name,object_type=typ,material=material,field=key,value=json.dumps(val, ensure_ascii=False),numeric_value=(val if isinstance(val, (int,float)) and not isinstance(val,bool) else '')))
            flat_by_scene[scene.name] = full
        rows.append(row)
    return rows, parameter_rows, object_rows, flat_by_scene

def create_reports(root, out):
    rows, pars, objs, flats = scan(root)
    save_csv(out/'scene_manifest.csv', rows, META)
    save_csv(out/'object_inventory.csv', objs, ('scene_id','parent_id','variant','object_name','object_type','material'))
    save_csv(out/'parameter_long.csv', pars, ('scene_id','parent_id','variant','object_name','object_type','material','field','value','numeric_value'))
    # Numeric distributions (presence and unique values are conditioned on field and material).
    by_key = defaultdict(list)
    frequency = Counter()
    for p in pars:
        key = (p['object_type'],p['material'],p['field'])
        frequency[(*key,p['value'])] += 1
        if isinstance(p['numeric_value'], (float,int)):
            by_key[key].append(float(p['numeric_value']))
    num_stats = []
    for key, vals in sorted(by_key.items()):
        vals.sort(); n=len(vals)
        def q(f): return vals[round((n-1)*f)]
        num_stats.append(dict(object_type=key[0],material=key[1],field=key[2],count=n,unique_values=len(set(vals)),min=vals[0],p05=q(.05),median=q(.5),p95=q(.95),max=vals[-1]))
    save_csv(out/'numeric_statistics.csv',num_stats,('object_type','material','field','count','unique_values','min','p05','median','p95','max'))
    freq_rows = [dict(object_type=k[0],material=k[1],field=k[2],value=k[3],count=n) for k,n in frequency.items()]
    save_csv(out/'value_frequencies.csv',sorted(freq_rows,key=lambda r:(r['field'],-r['count'])),('object_type','material','field','value','count'))
    # Compare related scenes to BASE using YAML paths; no claim that change is physically causal.
    base = {r['parent_id']:r['scene_id'] for r in rows if r['variant']=='base' and r['scene_id'] in flats}
    diffs=[]
    for r in rows:
        if r['variant']=='base' or r['scene_id'] not in flats or r['parent_id'] not in base: continue
        before=flats[base[r['parent_id']]]; after=flats[r['scene_id']]
        for field in sorted(before.keys() | after.keys()):
            a,b=before.get(field,'__MISSING__'),after.get(field,'__MISSING__')
            if a != b:
                diffs.append(dict(parent_id=r['parent_id'],scene_id=r['scene_id'],variant=r['variant'],field=field,base_value=json.dumps(a,ensure_ascii=False),variant_value=json.dumps(b,ensure_ascii=False)))
    save_csv(out/'family_differences.csv',diffs,('parent_id','scene_id','variant','field','base_value','variant_value'))
    overview = dict(scene_count=len(rows),families=len({r['parent_id'] for r in rows}),variants=dict(Counter(r['variant'] for r in rows)),missing_yaml=sum(not r['has_yaml'] for r in rows),missing_program=sum(not r['has_program'] for r in rows),missing_video=sum(not r['has_video'] for r in rows),invalid_yaml=sum(bool(r['yaml_error']) for r in rows),invalid_program_syntax=sum(r['program_syntax_ok'] is False for r in rows),objects_by_type=dict(Counter(o['object_type'] for o in objs)),materials=dict(Counter(o['material'] for o in objs)),program_unique_hashes=len({r['program_sha256'] for r in rows if r['program_sha256']}),family_differences=len(diffs))
    (out/'summary.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(overview,indent=2,ensure_ascii=False))
    print('Reports:',out.resolve())

if __name__ == '__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--root',type=Path,required=True,help='Root containing part_*/instances/scene_*')
    cli.add_argument('--out',type=Path,default=Path('data/metadata'))
    args=cli.parse_args()
    if not args.root.is_dir(): cli.error(f'Not a directory: {args.root}')
    create_reports(args.root,args.out)
