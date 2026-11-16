"""Read SFT-2000 canonical records and decode exact zero-based video frames."""

import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from PIL import Image


class Dataset:
    def __init__(self, path):
        self.root = Path(path)
        self.archive = None
        if self.root.is_file():
            self.archive = zipfile.ZipFile(self.root)
            index = "SFT-2000/canonical_records.jsonl"
            self.zip_prefix = "SFT-2000/" if index in self.archive.namelist() else ""
        elif not (self.root / "canonical_records.jsonl").is_file():
            self.root /= "SFT-2000"
        with self.open("canonical_records.jsonl") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.by_id = {record["scene_id"]: record for record in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Duplicate scene IDs in canonical_records.jsonl")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.archive:
            self.archive.close()

    @contextmanager
    def open(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Invalid dataset-relative path")
        if self.archive:
            member = self.archive.getinfo(self.zip_prefix + relative.as_posix())
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unexpected ZIP symlink")
            with self.archive.open(member) as stream:
                yield stream
        else:
            path = self.root.joinpath(*relative.parts).resolve()
            path.relative_to(self.root.resolve())
            with path.open("rb") as stream:
                yield stream

    def video_to(self, record, destination):
        """Extract one scene's video from a ZIP, or reuse a file from extracted data."""
        name = "raw/" + record["video_path"]
        with self.open(name) as stream:
            if not self.archive:
                return Path(stream.name)
            with Path(destination).open("xb") as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
        return Path(destination)

    def description(self, record):
        with self.open(record["scene_spec_path"]) as stream:
            value = json.load(stream)["scene"].get("description")
        return value[:1500] if isinstance(value, str) else ""


def protocol(record, override_dt=None):
    observations = record["observation"]
    indices = tuple(observations["frame_indices"])
    metadata = observations["video_metadata"]
    cutoff = int(observations["heldout_start_frame_index"])
    count = int(metadata["frame_count"])
    fps = float(metadata["fps"])
    if (len(indices) != 8 or tuple(sorted(set(indices))) != indices
            or not 0 <= indices[0] <= indices[-1] < cutoff < count <= 600
            or not math.isfinite(fps) or not 0 < fps <= 240):
        raise ValueError("Unsupported canonical observation protocol")
    times = observations.get("physical_timestamps_seconds")
    frame_dt, timing_source = 0.004, "approximation_default"
    if times and len(times) == len(indices):
        intervals = [(b - a) / (j - i) for a, b, i, j in zip(times, times[1:], indices, indices[1:])]
        estimate = statistics.median(intervals)
        if 0 < estimate <= 0.025:
            frame_dt, timing_source = estimate, "canonical_physical_timestamps"
    if override_dt is not None:
        if not 0 < override_dt <= 0.025:
            raise ValueError("frame_dt must be in (0, 0.025]")
        frame_dt, timing_source = override_dt, "explicit_override"
    return {
        "protocol": observations["protocol_version"], "indices": indices, "heldout_start": cutoff,
        "frame_count": count, "video_fps": fps, "frame_dt": frame_dt, "timing_source": timing_source,
    }


def ffmpeg():
    executable = os.environ.get("MPM_FFMPEG") or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("Install FFmpeg or set MPM_FFMPEG to its executable path")
    return executable


def read_frames(video, indices, resolution):
    """Decode in frame order instead of approximating frame numbers with timestamps."""
    selected = "+".join(f"eq(n\\,{index})" for index in indices)
    filters = f"select={selected},scale={resolution}:{resolution}:flags=area"
    command = [
        ffmpeg(), "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
        "-vf", filters, "-fps_mode", "passthrough", "-frames:v", str(len(indices)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True, timeout=90)
    frame_size = resolution * resolution * 3
    if len(result.stdout) != frame_size * len(indices):
        raise ValueError(f"Could not decode every requested frame from {Path(video).name}")
    return [Image.frombytes("RGB", (resolution, resolution), result.stdout[i:i + frame_size])
            for i in range(0, len(result.stdout), frame_size)]


def encode_video(directory, fps, destination):
    command = [ffmpeg(), "-nostdin", "-v", "error", "-n", "-framerate", str(fps),
               "-i", str(Path(directory) / "frame_%04d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(destination)]
    subprocess.run(command, check=True, timeout=120)
