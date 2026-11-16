"""Fixed, CPU-only 2D MLS-MPM solver for bounded model-generated scene parameters."""

import math
import os
import re
from pathlib import Path

import numpy as np
import taichi as ti
from PIL import Image, ImageColor, ImageDraw


MATERIALS = {"elastic": 0, "water": 1, "snow": 2, "jelly": 3}


def validate_scene(scene):
    def fields(obj, required, optional=()):
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object")
        if set(required) - obj.keys() or obj.keys() - set(required) - set(optional):
            raise ValueError(f"Expected fields {required}; optional fields {optional}")

    def scalar(value, low, high, label):
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"{label} must be a number in [{low}, {high}]")
        return float(value)

    def vector(values, size, low, high, label):
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"{label} must be an array of {size} values")
        return [scalar(value, low, high, label) for value in values]

    def color(value, label):
        if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            return value.lower()
        if (isinstance(value, list) and len(value) == 3
                and all(type(channel) is int and 0 <= channel <= 255 for channel in value)):
            return "#" + "".join(f"{channel:02x}" for channel in value)
        raise ValueError(f"{label} must be #RRGGBB or three integer RGB channels in [0,255]")

    fields(scene, ("gravity", "background", "bodies", "obstacles"))
    bodies = scene["bodies"]
    obstacles = scene["obstacles"]
    if not isinstance(bodies, list) or not 1 <= len(bodies) <= 4:
        raise ValueError("Supply one to four particle bodies")
    if not isinstance(obstacles, list) or len(obstacles) > 12:
        raise ValueError("Supply zero to twelve static obstacles")
    normalized_bodies = []
    for body in bodies:
        fields(body, ("shape", "geometry", "material", "color"), ("velocity", "stiffness", "density"))
        shape = body["shape"]
        material = body["material"]
        if shape not in ("box", "circle") or not isinstance(material, str) or material not in MATERIALS:
            raise ValueError("Bodies support box/circle and elastic/water/snow/jelly")
        geometry = vector(body["geometry"], 4 if shape == "box" else 3, 0.035, 0.965, "body geometry")
        if shape == "box":
            if geometry[2] - geometry[0] < 0.015 or geometry[3] - geometry[1] < 0.015:
                raise ValueError("Box geometry is [x_min,y_min,x_max,y_max] with positive extent")
        else:
            x, y, radius = geometry
            if min(x, y, 1 - x, 1 - y) < radius + 0.03:
                raise ValueError("Keep the full body circle inside [0.03,0.97]")
        normalized_bodies.append({
            "shape": shape, "geometry": geometry, "material": material, "color": color(body["color"], "Body color"),
            "velocity": vector(body.get("velocity", [0, 0]), 2, -5, 5, "initial velocity"),
            "stiffness": scalar(body.get("stiffness", 900 if material == "jelly" else 1800),
                                50, 4000, "stiffness"),
            "density": scalar(body.get("density", 1), 0.5, 4, "density"),
        })
    normalized_obstacles = []
    for obstacle in obstacles:
        fields(obstacle, ("shape", "geometry", "color"))
        shape = obstacle["shape"]
        if shape not in ("segment", "circle"):
            raise ValueError("Static obstacles support segment and circle")
        geometry = vector(obstacle["geometry"], 5 if shape == "segment" else 3, 0, 1, "obstacle geometry")
        scalar(geometry[-1], 0.002, 0.40, "obstacle radius")
        if shape == "segment" and geometry[:2] == geometry[2:4]:
            raise ValueError("Segment endpoints must differ")
        normalized_obstacles.append({"shape": shape, "geometry": geometry,
                                     "color": color(obstacle["color"], "Obstacle color")})
    gravity = scene["gravity"]
    if type(gravity) in (int, float):
        gravity = [0, -abs(gravity)]
    return {"gravity": vector(gravity, 2, -40, 40, "gravity"),
            "background": color(scene["background"], "Background color"), "bodies": normalized_bodies,
            "obstacles": normalized_obstacles}


@ti.data_oriented
class Solver:
    def __init__(self, scene, seed=17):
        self.scene = validate_scene(scene)
        self.n = 64
        self.dx = 1.0 / self.n
        generator = np.random.default_rng(seed)
        positions, velocities, kinds, masses, volumes, stiffnesses, colors = [], [], [], [], [], [], []
        for body in self.scene["bodies"]:
            bounds = body["geometry"]
            if body["shape"] == "box":
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                count = int(np.clip(area * 36000, 450, 4500))
                samples = generator.uniform(bounds[:2], bounds[2:], size=(count, 2))
            else:
                area = math.pi * bounds[2] ** 2
                count = int(np.clip(area * 36000, 450, 4500))
                angles = generator.uniform(0, 2 * math.pi, count)
                radii = np.sqrt(generator.random(count)) * bounds[2]
                samples = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None] + bounds[:2]
            volume = area / count
            positions.extend(samples)
            velocities.extend([body["velocity"]] * count)
            kinds.extend([MATERIALS[body["material"]]] * count)
            volumes.extend([volume] * count)
            masses.extend([volume * body["density"]] * count)
            stiffnesses.extend([body["stiffness"]] * count)
            colors.extend([ImageColor.getrgb(body["color"])] * count)
        size = len(positions)
        self.x = ti.Vector.field(2, ti.f32, size)
        self.v = ti.Vector.field(2, ti.f32, size)
        self.c = ti.Matrix.field(2, 2, ti.f32, size)
        self.f = ti.Matrix.field(2, 2, ti.f32, size)
        self.jp = ti.field(ti.f32, size)
        self.kind = ti.field(ti.i32, size)
        self.mass = ti.field(ti.f32, size)
        self.volume = ti.field(ti.f32, size)
        self.stiffness = ti.field(ti.f32, size)
        self.grid_m = ti.field(ti.f32, (self.n, self.n))
        self.grid_v = ti.Vector.field(2, ti.f32, (self.n, self.n))
        self.gravity = ti.Vector(self.scene["gravity"])
        self.colors = np.array(colors, dtype=np.uint8)
        self.obstacle_count = len(self.scene["obstacles"])
        self.colliders = ti.Vector.field(5, ti.f32, max(1, self.obstacle_count))
        for index, obstacle in enumerate(self.scene["obstacles"]):
            shape = obstacle["geometry"]
            if obstacle["shape"] == "circle":
                shape = shape[:2] + shape[:2] + shape[2:]
            self.colliders[index] = shape
        self.x.from_numpy(np.asarray(positions, dtype=np.float32))
        self.v.from_numpy(np.asarray(velocities, dtype=np.float32))
        self.kind.from_numpy(np.asarray(kinds, dtype=np.int32))
        self.mass.from_numpy(np.asarray(masses, dtype=np.float32))
        self.volume.from_numpy(np.asarray(volumes, dtype=np.float32))
        self.stiffness.from_numpy(np.asarray(stiffnesses, dtype=np.float32))
        self.initialize_deformation()

    @ti.kernel
    def initialize_deformation(self):
        for p in self.x:
            self.f[p] = ti.Matrix.identity(ti.f32, 2)
            self.jp[p] = 1

    @ti.kernel
    def substep(self, dt: ti.f32):
        for i, j in self.grid_m:
            self.grid_v[i, j] = ti.Vector.zero(ti.f32, 2)
            self.grid_m[i, j] = 0
        for p in self.x:
            p_grid = self.x[p] * self.n
            base = ti.floor(p_grid - 0.5).cast(ti.i32)
            frac = p_grid - base.cast(ti.f32)
            weights = [0.5 * (1.5 - frac) ** 2, 0.75 - (frac - 1) ** 2, 0.5 * (frac - 0.5) ** 2]
            identity = ti.Matrix.identity(ti.f32, 2)
            deformation = (identity + dt * self.c[p]) @ self.f[p]
            mu = self.stiffness[p] / 2.4
            lam = self.stiffness[p] / 3.6
            u, sig, v = ti.svd(deformation)
            jacobian = 1.0
            for axis in ti.static(range(2)):
                sigma = ti.max(0.05, sig[axis, axis])
                limited = ti.min(4.0, sigma)
                if self.kind[p] == 2 or self.kind[p] == 3:
                    lo, hi = 0.975, 1.0045
                    if self.kind[p] == 3:
                        lo, hi = 0.92, 1.09
                    limited = ti.math.clamp(sigma, lo, hi)
                    self.jp[p] *= sigma / limited
                sig[axis, axis] = limited
                jacobian *= limited
            self.jp[p] = ti.math.clamp(self.jp[p], 0.3, 2.0)
            if self.kind[p] == 2:
                hardening = ti.math.clamp(ti.exp(10 * (1 - self.jp[p])), 0.15, 4.0)
                mu *= hardening
                lam *= hardening
            deformation = u @ sig @ v.transpose()
            if self.kind[p] == 1:
                mu = 0
                deformation = identity * ti.sqrt(jacobian)
            self.f[p] = deformation
            stress = 2 * mu * (deformation - u @ v.transpose()) @ deformation.transpose()
            stress += identity * lam * jacobian * (jacobian - 1)
            affine = self.mass[p] * self.c[p] - dt * 4 * self.n * self.n * self.volume[p] * stress
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                distance = (offset.cast(ti.f32) - frac) * self.dx
                weight = weights[i][0] * weights[j][1]
                self.grid_v[base + offset] += weight * (self.mass[p] * self.v[p] + affine @ distance)
                self.grid_m[base + offset] += weight * self.mass[p]
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:
                velocity = self.grid_v[i, j] / self.grid_m[i, j] + dt * self.gravity
                if (i < 3 and velocity[0] < 0) or (i >= self.n - 3 and velocity[0] > 0):
                    velocity[0] = 0
                if (j < 3 and velocity[1] < 0) or (j >= self.n - 3 and velocity[1] > 0):
                    velocity[1] = 0
                point = ti.Vector([i, j]) * self.dx
                for k in range(self.obstacle_count):
                    collider = self.colliders[k]
                    origin = ti.Vector([collider[0], collider[1]])
                    line = ti.Vector([collider[2], collider[3]]) - origin
                    along = ti.math.clamp((point - origin).dot(line) / (line.dot(line) + 1e-12), 0.0, 1.0)
                    offset = point - (origin + along * line)
                    distance = offset.norm()
                    if 0 < distance < collider[4] + 0.65 * self.dx:
                        normal = offset / distance
                        velocity -= 1.1 * ti.min(0.0, velocity.dot(normal)) * normal
                self.grid_v[i, j] = velocity
        for p in self.x:
            p_grid = self.x[p] * self.n
            base = ti.floor(p_grid - 0.5).cast(ti.i32)
            frac = p_grid - base.cast(ti.f32)
            weights = [0.5 * (1.5 - frac) ** 2, 0.75 - (frac - 1) ** 2, 0.5 * (frac - 0.5) ** 2]
            velocity = ti.Vector.zero(ti.f32, 2)
            affine = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                distance = offset.cast(ti.f32) - frac
                weight = weights[i][0] * weights[j][1]
                gv = self.grid_v[base + offset]
                velocity += weight * gv
                affine += weight * 4 * self.n * gv.outer_product(distance)
            self.v[p] = velocity
            self.c[p] = affine
            self.x[p] = ti.math.clamp(self.x[p] + dt * velocity, 2 * self.dx, 1 - 2 * self.dx)

    def render(self, resolution):
        positions = self.x.to_numpy()
        if not np.isfinite(positions).all():
            raise ValueError("Solver generated non-finite particle positions")
        background = ImageColor.getrgb(self.scene["background"])
        frame = np.full((resolution, resolution, 3), background, dtype=np.uint8)
        x = np.rint(positions[:, 0] * (resolution - 1)).astype(int)
        y = np.rint((1 - positions[:, 1]) * (resolution - 1)).astype(int)
        for ox, oy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            frame[np.clip(y + oy, 0, resolution - 1), np.clip(x + ox, 0, resolution - 1)] = self.colors
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        size = resolution - 1
        for obstacle in self.scene["obstacles"]:
            geometry = obstacle["geometry"]
            px, py = geometry[0] * size, (1 - geometry[1]) * size
            r = geometry[-1] * size
            color = obstacle["color"]
            if obstacle["shape"] == "circle":
                draw.ellipse((px - r, py - r, px + r, py + r), fill=color)
            else:
                qx, qy = geometry[2] * size, (1 - geometry[3]) * size
                draw.line((px, py, qx, qy), fill=color, width=max(1, round(2 * r)))
        return image


def simulate(scene, indices, *, frame_dt, resolution=256, seed=17, frames_dir=None):
    """Return the requested indices. Optionally save every frame through the last index."""
    if not 0 < frame_dt <= 0.025 or not 64 <= resolution <= 512:
        raise ValueError("Unsupported physics frame interval or rendering size")
    indices = tuple(int(index) for index in indices)
    if not indices or tuple(sorted(set(indices))) != indices or not 0 <= indices[0] <= indices[-1] < 600:
        raise ValueError("Expected ordered, unique frame indices inside 0–599")
    normalized = validate_scene(scene)
    cache = Path(os.environ.get("MPM_CACHE_DIR", ".mpm_taichi_cache")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.cpu, cpu_max_num_threads=4, offline_cache_file_path=str(cache), random_seed=seed)
    try:
        solver = Solver(normalized, seed)
        stable_dt = 0.00016
        substeps = max(1, math.ceil(frame_dt / stable_dt))
        dt = frame_dt / substeps
        requested = set(indices)
        frames = []
        if frames_dir:
            frames_dir = Path(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=False)
        for frame_number in range(indices[-1] + 1):
            if frame_number in requested or frames_dir:
                image = solver.render(resolution)
                if frames_dir:
                    image.save(frames_dir / f"frame_{frame_number:04d}.png")
                if frame_number in requested:
                    frames.append(image)
            if frame_number < indices[-1]:
                for _ in range(substeps):
                    solver.substep(dt)
        return frames
    finally:
        ti.reset()
