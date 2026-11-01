"""Self-contained, read-only MPMWorlds preprocessing and split pipeline."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCENE_NAME = re.compile(r"^scene_(?P<parent_id>\d+)_(?P<variant>base|modification|perturbation)(?:_(?P<variant_id>\d+))?$")
SPLITS = ("train", "validation", "test")
TARGET_NAMES = (
    "constitutive_model",
    "density",
    "youngs_modulus",
    "poissons_ratio",
    "fluid_bulk_modulus",
    "initial_velocity",
    "angular_velocity",
)
PROMPTS = {
    "video_to_code": {
        "version": "m2-video-to-code-v1",
        "text": "Given the ordered video observations, produce one complete executable Python Taichi MPM program. Return code only.",
    },
    "video_to_scene": {
        "version": "m2-video-to-scene-v1",
        "text": "Given the ordered video observations, produce the Canonical Scene Spec JSON. Return JSON only.",
    },
}


@dataclass
class Scene:
    scene_id: str
    parent_id: str
    variant: str
    variant_id: str
    part: str
    relative_path: str
    directory: Path
    config: dict[str, Any] | None
    yaml_hash: str
    program_hash: str
    normalised_program_hash: str
    valid_yaml: bool
    valid_program: bool
    has_video: bool
    reasons: list[str]


class _NormaliseConstants(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) or node.value is None:
            value: Any = node.value
        elif isinstance(node.value, (int, float, complex)):
            value = 0
        elif isinstance(node.value, str):
            value = ""
        elif isinstance(node.value, bytes):
            value = b""
        else:
            value = None
        return ast.copy_location(ast.Constant(value=value), node)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _discover(data_root: Path) -> list[Scene]:
    scenes: list[Scene] = []
    for directory in sorted(data_root.rglob("scene_*"), key=lambda item: item.as_posix()):
        if not directory.is_dir() or directory.parent.name != "instances":
            continue
        match = SCENE_NAME.fullmatch(directory.name)
        if match is None:
            continue
        relative = directory.relative_to(data_root).as_posix()
        part = next((parent.name for parent in directory.parents if parent.parent == data_root), "")
        yaml_path, program_path, video_path = directory / "scene.yaml", directory / "program.py", directory / "video.mp4"
        config: dict[str, Any] | None = None
        yaml_hash = ""
        program_hash = ""
        normalised_program_hash = ""
        reasons: list[str] = []
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("yaml_root_not_mapping")
            yaml_hash = _sha256(yaml_path)
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            reasons.append("yaml_missing_or_invalid")
            config = None
        try:
            source = program_path.read_bytes()
            program_hash = hashlib.sha256(source).hexdigest()
            tree = ast.parse(source.decode("utf-8"), filename=program_path.name)
            normalised_program_hash = hashlib.sha256(
                ast.dump(
                    _NormaliseConstants().visit(ast.fix_missing_locations(tree)),
                    annotate_fields=False,
                    include_attributes=False,
                ).encode("utf-8")
            ).hexdigest()
        except (OSError, UnicodeDecodeError, SyntaxError):
            reasons.append("program_missing_or_invalid_ast")
        if not video_path.is_file():
            reasons.append("video_missing")
        groups = match.groupdict()
        scenes.append(
            Scene(
                scene_id=directory.name,
                parent_id=groups["parent_id"],
                variant=groups["variant"],
                variant_id=groups["variant_id"] or "",
                part=part,
                relative_path=relative,
                directory=directory,
                config=config,
                yaml_hash=yaml_hash,
                program_hash=program_hash,
                normalised_program_hash=normalised_program_hash,
                valid_yaml=config is not None,
                valid_program=bool(normalised_program_hash),
                has_video=video_path.is_file(),
                reasons=reasons,
            )
        )
    if not scenes:
        raise ValueError(f"No valid MPMWorlds scene directories under {data_root}")
    return scenes


def _mark_duplicate_scene_ids(scenes: list[Scene]) -> None:
    by_id: dict[str, list[Scene]] = defaultdict(list)
    for scene in scenes:
        by_id[scene.scene_id].append(scene)
    for copies in by_id.values():
        if len(copies) < 2:
            continue
        fingerprints: list[tuple[str, str, str]] = []
        for scene in copies:
            video = scene.directory / "video.mp4"
            fingerprints.append((scene.yaml_hash, scene.program_hash, _sha256(video) if video.is_file() else ""))
        if fingerprints and all(fingerprint == fingerprints[0] and all(fingerprint) for fingerprint in fingerprints):
            canonical = min(copies, key=lambda scene: scene.relative_path)
            for scene in copies:
                if scene is not canonical:
                    scene.reasons.append("duplicate_scene_id_noncanonical_copy")
        else:
            for scene in copies:
                scene.reasons.append("duplicate_scene_id_conflict")


def _mark_cross_family_programs(scenes: list[Scene]) -> None:
    for attribute in ("program_hash", "normalised_program_hash"):
        groups: dict[str, list[Scene]] = defaultdict(list)
        for scene in scenes:
            value = getattr(scene, attribute)
            if value and scene.valid_program:
                groups[value].append(scene)
        for matches in groups.values():
            if len({scene.parent_id for scene in matches}) > 1:
                for scene in matches:
                    scene.reasons.append("cross_family_program_similarity")


def _parent_splits(scenes: list[Scene], seed: int) -> dict[str, str]:
    parents = sorted({scene.parent_id for scene in scenes})
    random.Random(seed).shuffle(parents)
    train_end = round(len(parents) * 0.8)
    validation_end = train_end + round(len(parents) * 0.1)
    return {
        parent: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, parent in enumerate(parents)
    }


def _rate(value: str) -> float | None:
    try:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator)
        return result if math.isfinite(result) else None
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def _probe_video(scene: Scene) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required to build frame-index manifests")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,width,height,duration:format=duration",
            "-of",
            "json",
            str(scene.directory / "video.mp4"),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("ffprobe_failed")
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    try:
        frame_count = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ffprobe_missing_exact_frame_count") from error
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    return {
        "frame_count": frame_count,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": _rate(stream.get("avg_frame_rate", "")),
        "duration_seconds": float(duration) if duration not in (None, "") else None,
    }


def _probe_or_error(scene: Scene) -> dict[str, Any] | Exception:
    try:
        return _probe_video(scene)
    except Exception as error:
        return error


def _observation(metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    total = int(metadata["frame_count"])
    observed_end = math.ceil(total * 0.70)
    indices = [math.floor((index + 0.5) * observed_end / 8) for index in range(8)]
    if total < 8 or observed_end >= total or len(set(indices)) != 8 or any(index >= observed_end for index in indices):
        raise ValueError("invalid_video_length_for_observation_protocol")
    simulation = config.get("simulation") if isinstance(config.get("simulation"), dict) else {}
    dt = simulation.get("time_step_dt")
    substeps = simulation.get("substeps_per_frame")
    seconds_per_frame = float(dt) * float(substeps) if isinstance(dt, (int, float)) and isinstance(substeps, (int, float)) else None
    return {
        "protocol_version": "m2-eight-first70-bin-centers-v1",
        "frame_indices": indices,
        "observed_frame_count": 8,
        "observed_fraction": 0.70,
        "heldout_start_frame_index": observed_end,
        "physical_timestamps_seconds": [index * seconds_per_frame for index in indices] if seconds_per_frame is not None else None,
        "physical_timestamp_status": "yaml_dt_times_substeps" if seconds_per_frame is not None else "not_available",
        "video_metadata": metadata,
    }


def _object_identity(raw: dict[str, Any], index: int) -> tuple[str, str]:
    return str(raw.get("name") or f"object_{index}"), str(raw.get("object_type") or raw.get("type") or raw.get("body_type") or "unknown")


def _leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [item for key in sorted(value, key=str) for item in _leaves(value[key], f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return [item for index, child in enumerate(value) for item in _leaves(child, f"{prefix}[{index}]")]
    return [(prefix, value)]


def _object_sections(raw: dict[str, Any]) -> dict[str, Any]:
    remaining = {key: deepcopy(value) for key, value in raw.items() if key not in {"name", "object_type", "type", "body_type"}}

    def take(*keys: str) -> dict[str, Any]:
        return {key: remaining.pop(key) for key in keys if key in remaining}

    material = take("physics_properties", "material_properties", "material")
    collision = take("collision", "contact", "collider")
    contact_values = {
        key: value
        for values in material.values()
        if isinstance(values, dict)
        for key, value in values.items()
        if any(token in key.lower() for token in ("friction", "collision", "contact", "restitution"))
    }
    if contact_values:
        collision["physics_contact_properties"] = contact_values
    return {
        "geometry": take("shape", "geometry", "mesh"),
        "material": material,
        "initial_conditions": take("initial_state", "initial_conditions", "initialization"),
        "kinematics": take("kinematics", "motion"),
        "collision_contact": collision,
        "rendering": take("visualization", "rendering"),
        "other_reconstruction_fields": remaining,
    }


def _scene_spec(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    objects = config.get("objects") if isinstance(config.get("objects"), list) else []
    object_specs: list[dict[str, Any]] = []
    coverage: list[dict[str, str]] = []
    for index, raw in enumerate(objects):
        if not isinstance(raw, dict):
            continue
        name, object_type = _object_identity(raw, index)
        labels: dict[str, Any] = {}
        leaves = _leaves(raw)
        for target in TARGET_NAMES:
            match = next(((path, value) for path, value in leaves if path.rsplit(".", 1)[-1].lower() == target), None)
            if match is None:
                continue
            path, value = match
            labels[target] = {"value": value, "quality_status": "unverified", "yaml_field_path": f"objects.{name}.{path}"}
            coverage.append({"target": target, "quality_status": "unverified"})
        stiffness = {target: labels[target] for target in ("youngs_modulus", "poissons_ratio") if target in labels}
        if stiffness:
            labels["effective_stiffness_inputs"] = stiffness
            coverage.append({"target": "effective_stiffness_inputs", "quality_status": "unverified"})
        object_specs.append({"name": name, "object_type": object_type, **_object_sections(raw), "physical_target_labels": labels})
    top_level = {key: deepcopy(value) for key, value in config.items() if key not in {"objects", "simulation", "environment", "output", "description"}}
    return {
        "schema_version": "canonical-scene-spec-v0.1",
        "scene": {
            "description": config.get("description"),
            "dimension": (config.get("simulation") or {}).get("dimension") if isinstance(config.get("simulation"), dict) else None,
            "objects": object_specs,
            "environment": deepcopy(config.get("environment", {})),
            "solver_settings": deepcopy(config.get("simulation", {})),
            "output_settings": deepcopy(config.get("output", {})),
            "other_reconstruction_fields": top_level,
        },
        "target_policy": {
            "primary": ["constitutive_model", "geometry_layout", "initial_velocity", "angular_velocity_when_present"],
            "auxiliary": ["density_conditioned_on_material", "youngs_modulus_and_poissons_ratio_as_effective_stiffness_inputs", "fluid_bulk_modulus_when_present"],
            "label_quality": "Raw YAML labels are retained but unverified; this pipeline never executes programs or infers semantic equivalence.",
        },
    }, coverage


def _sft_records(canonical: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    common = {
        "record_version": "m2-sft-manifest-v1",
        "scene_id": canonical["scene_id"],
        "parent_id": canonical["parent_id"],
        "split": canonical["split"],
        "model_input": {"video_path": canonical["video_path"], "observation": canonical["observation"]},
        "label_quality": canonical["label_quality"],
        "source_provenance": canonical["source_provenance"],
    }
    code = {
        **common,
        "record_id": f"{canonical['scene_id']}:video_to_code",
        "task_type": "video_to_code",
        "prompt_ref": PROMPTS["video_to_code"]["version"],
        "target": {"target_type": "python_program", "relative_path": canonical["program_path"]},
        "sampling_weight": 1.0,
    }
    scene = {
        **common,
        "record_id": f"{canonical['scene_id']}:video_to_scene",
        "task_type": "video_to_scene",
        "prompt_ref": PROMPTS["video_to_scene"]["version"],
        "target": {"target_type": "canonical_scene_spec", "relative_path": canonical["scene_spec_path"]},
        "sampling_weight": 1.0,
    }
    return {
        "code_only": [code],
        "scene_only": [scene],
        "physics_aware": [
            {**scene, "record_id": f"{canonical['scene_id']}:physics_aware:video_to_scene", "sampling_weight": 0.5, "multitask_group": canonical["scene_id"]},
            {**code, "record_id": f"{canonical['scene_id']}:physics_aware:video_to_code", "sampling_weight": 0.5, "multitask_group": canonical["scene_id"]},
        ],
    }


def _validate(canonical: list[dict[str, Any]], tasks: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    errors: list[str] = []
    seen_paths: set[str] = set()
    parent_splits: dict[str, set[str]] = defaultdict(set)
    for record in canonical:
        path = record["source_provenance"]["dataset_relative_path"]
        if path in seen_paths:
            errors.append(f"duplicate_record:{path}")
        seen_paths.add(path)
        parent_splits[record["parent_id"]].add(record["split"])
        if any(record[key].startswith("/") for key in ("video_path", "program_path", "scene_spec_path")):
            errors.append(f"absolute_path:{path}")
    for parent, splits in parent_splits.items():
        if len(splits) != 1:
            errors.append(f"parent_overlap:{parent}")
    expected = {"code_only": len(canonical), "scene_only": len(canonical), "physics_aware": 2 * len(canonical)}
    actual = {strategy: sum(len(rows) for rows in by_split.values()) for strategy, by_split in tasks.items()}
    if actual != expected:
        errors.append("sft_record_count_mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "record_count": len(canonical),
        "scene_counts": dict(sorted(Counter(record["split"] for record in canonical).items())),
        "parent_count": len(parent_splits),
        "parent_overlap_count": sum(len(splits) != 1 for splits in parent_splits.values()),
        "sft_record_counts": actual,
    }


def _subset_targets(sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    weights = {"train": 0.8, "validation": 0.1, "test": 0.1}
    targets = {split: math.floor(sample_size * weight) for split, weight in weights.items()}
    remainder = sample_size - sum(targets.values())
    fractions = sorted(SPLITS, key=lambda split: (sample_size * weights[split] - targets[split], -SPLITS.index(split)), reverse=True)
    for split in fractions[:remainder]:
        targets[split] += 1
    return targets


def _select_parent_groups(records: list[dict[str, Any]], target: int, split: str, seed: int) -> set[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["parent_id"]].append(record)
    ordered = sorted(
        groups,
        key=lambda parent: hashlib.sha256(f"{seed}:sft-subset-v1:{split}:{parent}".encode("utf-8")).hexdigest(),
    )
    previous: dict[int, tuple[int, str]] = {0: (-1, "")}
    for parent in ordered:
        size = len(groups[parent])
        for subtotal in sorted(tuple(previous), reverse=True):
            candidate = subtotal + size
            if candidate <= target and candidate not in previous:
                previous[candidate] = (subtotal, parent)
    if target not in previous:
        available = sum(len(group) for group in groups.values())
        raise ValueError(f"Cannot form an exact {target}-record {split} subset from complete parent groups; {available} records available.")
    selected: set[str] = set()
    subtotal = target
    while subtotal:
        subtotal, parent = previous[subtotal]
        selected.add(parent)
    return selected


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Non-portable path in source manifest: {relative}")
    child = (root / path).resolve()
    if root != child and root not in child.parents:
        raise ValueError(f"Source manifest path escapes its root: {relative}")
    return child


def build_sft_subset(source_root: Path, output_root: Path, sample_size: int, seed: int = 20261025, overwrite: bool = False) -> dict[str, Any]:
    """Create an exact-size, family-complete SFT subset from a full M2 dataset."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_records = [json.loads(line) for line in (source_root / "canonical_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not source_records:
        raise ValueError(f"No canonical records in {source_root}")
    targets = _subset_targets(sample_size)
    selected: list[dict[str, Any]] = []
    selected_parent_counts: dict[str, int] = {}
    for split in SPLITS:
        candidates = [record for record in source_records if record.get("split") == split]
        parents = _select_parent_groups(candidates, targets[split], split, seed)
        selected.extend(record for record in candidates if record["parent_id"] in parents)
        selected_parent_counts[split] = len(parents)
    selected.sort(key=lambda record: record["source_provenance"]["dataset_relative_path"])
    if len(selected) != sample_size:
        raise RuntimeError(f"Subset selection returned {len(selected)} records, expected {sample_size}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        tasks: dict[str, dict[str, list[dict[str, Any]]]] = {strategy: {split: [] for split in SPLITS} for strategy in ("code_only", "scene_only", "physics_aware")}
        for record in selected:
            spec_path = record.get("scene_spec_path", "")
            source_spec = _safe_child(source_root, spec_path)
            if not source_spec.is_file():
                raise ValueError(f"Scene Spec is missing from source dataset: {spec_path}")
            destination_spec = _safe_child(staging, spec_path)
            destination_spec.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_spec, destination_spec)
            for strategy, task_records in _sft_records(record).items():
                tasks[strategy][record["split"]].extend(task_records)
        validation = _validate(selected, tasks)
        if not validation["valid"] or validation["scene_counts"] != targets:
            raise RuntimeError(f"Subset validation failed: {validation}")
        _write_jsonl(staging / "canonical_records.jsonl", selected)
        _write_json(staging / "sft" / "prompts.json", PROMPTS)
        for strategy, by_split in tasks.items():
            for split in SPLITS:
                _write_jsonl(staging / "sft" / strategy / f"{split}.jsonl", by_split[split])
        selection = {
            "strategy": "deterministic_complete_parent_groups_exact_subset_sum",
            "seed": seed,
            "requested_record_count": sample_size,
            "source_record_count": len(source_records),
            "split_targets": targets,
            "selected_parent_counts": selected_parent_counts,
        }
        _write_json(staging / "reports" / "subset_selection.json", selection)
        _write_json(staging / "reports" / "split_validation.json", validation)
        summary = {
            "pipeline_version": "m2-sft-subset-v1",
            "canonical_record_count": len(selected),
            "split_counts": validation["scene_counts"],
            "sft_record_counts": validation["sft_record_counts"],
            "selection": selection,
            "safety": {"programs_executed": False, "frames_decoded": False, "models_loaded": False, "training_run": False},
        }
        _write_json(staging / "reports" / "data_summary.json", summary)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        return {**summary, "output_root": str(output_root)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_dataset(data_root: Path, output_root: Path, seed: int = 20261025, workers: int = 8, overwrite: bool = False) -> dict[str, Any]:
    """Process raw scenes into split, model-independent training manifests."""
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    scenes = _discover(data_root)
    _mark_duplicate_scene_ids(scenes)
    _mark_cross_family_programs(scenes)
    splits = _parent_splits(scenes, seed)
    selected = [scene for scene in scenes if not scene.reasons]
    selected.sort(key=lambda scene: scene.relative_path)
    probes: dict[str, dict[str, Any] | Exception] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for scene, result in zip(selected, executor.map(_probe_or_error, selected)):
            probes[scene.relative_path] = result
    usable: list[Scene] = []
    exclusions: list[dict[str, str]] = []
    for scene in scenes:
        reasons = list(dict.fromkeys(scene.reasons))
        probe = probes.get(scene.relative_path)
        if isinstance(probe, Exception):
            reasons.append(f"video_probe_failed:{type(probe).__name__}")
        if reasons:
            exclusions.append({"relative_path": scene.relative_path, "scene_id": scene.scene_id, "parent_id": scene.parent_id, "split": splits[scene.parent_id], "reason": ";".join(reasons)})
        else:
            usable.append(scene)
    if not usable:
        raise ValueError("No usable scenes after source validation")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        canonical: list[dict[str, Any]] = []
        tasks: dict[str, dict[str, list[dict[str, Any]]]] = {strategy: {split: [] for split in SPLITS} for strategy in ("code_only", "scene_only", "physics_aware")}
        coverage: Counter[tuple[str, str]] = Counter()
        for scene in sorted(usable, key=lambda item: item.relative_path):
            assert scene.config is not None
            metadata = probes[scene.relative_path]
            assert isinstance(metadata, dict)
            spec, labels = _scene_spec(scene.config)
            spec_path = f"scene_specs/{scene.scene_id}.json"
            _write_json(staging / spec_path, spec)
            canonical_record = {
                "record_version": "m2-canonical-record-v2",
                "scene_id": scene.scene_id,
                "parent_id": scene.parent_id,
                "variant": scene.variant,
                "split": splits[scene.parent_id],
                "video_path": f"{scene.relative_path}/video.mp4",
                "program_path": f"{scene.relative_path}/program.py",
                "scene_spec_path": spec_path,
                "observation": _observation(metadata, scene.config),
                "label_quality": {"status": "unverified_raw_yaml", "unverified_target_names": [label["target"] for label in labels]},
                "source_provenance": {
                    "dataset_relative_path": scene.relative_path,
                    "yaml_sha256": scene.yaml_hash,
                    "program_sha256": scene.program_hash,
                },
            }
            canonical.append(canonical_record)
            for label in labels:
                coverage[(label["target"], label["quality_status"])] += 1
            for strategy, records in _sft_records(canonical_record).items():
                tasks[strategy][canonical_record["split"]].extend(records)
        validation = _validate(canonical, tasks)
        if not validation["valid"]:
            raise RuntimeError(f"Dataset validation failed: {validation['errors']}")
        _write_jsonl(staging / "canonical_records.jsonl", canonical)
        _write_json(staging / "sft" / "prompts.json", PROMPTS)
        for strategy, by_split in tasks.items():
            for split in SPLITS:
                _write_jsonl(staging / "sft" / strategy / f"{split}.jsonl", by_split[split])
        _write_csv(staging / "reports" / "exclusions.csv", exclusions, ("relative_path", "scene_id", "parent_id", "split", "reason"))
        _write_csv(
            staging / "reports" / "label_coverage.csv",
            [{"target": target, "quality_status": status, "label_count": count} for (target, status), count in sorted(coverage.items())],
            ("target", "quality_status", "label_count"),
        )
        _write_json(staging / "reports" / "split_validation.json", validation)
        summary = {
            "pipeline_version": "m2-standalone-data-pipeline-v2",
            "raw_scene_count": len(scenes),
            "canonical_record_count": len(canonical),
            "excluded_record_count": len(exclusions),
            "split_seed": seed,
            "split_counts": validation["scene_counts"],
            "sft_record_counts": validation["sft_record_counts"],
            "source_root": data_root.name,
            "safety": {"programs_executed": False, "frames_decoded": False, "models_loaded": False, "training_run": False},
        }
        _write_json(staging / "reports" / "data_summary.json", summary)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        return {**summary, "output_root": str(output_root)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
