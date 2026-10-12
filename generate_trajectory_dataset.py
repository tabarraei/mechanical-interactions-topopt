#!/usr/bin/env python
from __future__ import annotations

import json
import os
import pickle
import statistics
import sys
import time
from typing import Any, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "99_shared_utilities"))
sys.path.insert(0, os.path.join(_ROOT, "05_self_supervised"))

from inr_render import remesh_from_coords
from topo_transformer import get_problem_loader
from topopt_utils.topopt_2d import optimize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "simp_trajectory")

NUM_SAMPLES = 2
SAMPLES_PER_SHARD = 1000
BATCH_SIZE = 32
SEED = 0
WIDTH = 1.0
HEIGHT = 2.0
RANDOM_ASPECT = False
NELS = 64
PENAL = 3.0
RMIN = 2.0
MAXITER = 100
TOLERANCE = 0.01
STORE_TRAJECTORY = True
OVERWRITE = False

SHARD_TEMPLATE = "topopt_samples_{index:05d}.pkl"
MANIFEST_NAME = "manifest.json"


def shard_filename(index: int) -> str:
    return SHARD_TEMPLATE.format(index=index)


def shard_paths(output_dir: str) -> list[str]:
    manifest = read_manifest(output_dir)
    if manifest is None:
        return []
    return [os.path.join(output_dir, row["file"]) for row in manifest["shards"]]


def read_manifest(output_dir: str) -> Optional[dict[str, Any]]:
    path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_atomic(path: str, dump) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        dump(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_shard(path: str, samples: list) -> None:
    _write_atomic(path, lambda f: pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL))


def write_manifest(output_dir: str, manifest: dict[str, Any]) -> None:
    path = os.path.join(output_dir, MANIFEST_NAME)
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    _write_atomic(path, lambda f: f.write(payload))


def completed_count(manifest: Optional[dict[str, Any]]) -> int:
    if not manifest:
        return 0
    return int(sum(row["num_samples"] for row in manifest["shards"]))


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _number_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    parts = [
        f"n={len(values)}",
        f"mean={statistics.mean(values):.1f}",
        f"median={statistics.median(values):.1f}",
        f"min={min(values):.0f}",
        f"max={max(values):.0f}",
    ]
    if len(values) > 1:
        parts.insert(2, f"std={statistics.stdev(values):.1f}")
    return " ".join(parts)


def _time_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    parts = [
        f"n={len(values)}",
        f"total={_fmt_seconds(sum(values))}",
        f"mean={_fmt_seconds(statistics.mean(values))}",
        f"median={_fmt_seconds(statistics.median(values))}",
        f"min={_fmt_seconds(min(values))}",
        f"max={_fmt_seconds(max(values))}",
    ]
    if len(values) > 1:
        parts.insert(3, f"std={_fmt_seconds(statistics.stdev(values))}")
    return " ".join(parts)


def _print_run_stats(completed: int, wall_times: list[float], iter_counts: list[int]) -> None:
    remaining = max(0, NUM_SAMPLES - completed)
    eta = statistics.mean(wall_times) * remaining if wall_times else 0.0
    print(
        f"[{completed}/{NUM_SAMPLES}] wall {_time_stats(wall_times)} eta={_fmt_seconds(eta)}",
        flush=True,
    )
    if iter_counts:
        print(f"[{completed}/{NUM_SAMPLES}] iters {_number_stats([float(n) for n in iter_counts])}", flush=True)


def trajectory_upper_bound_bytes(num_samples: int, nels: int, maxiter: int) -> int:
    frame = int(nels) * int(nels) * 8
    per_sample = maxiter * frame + frame + (nels + 1) * (nels + 1) * 2 * 8
    return int(num_samples) * per_sample


def _clear_output(output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name == MANIFEST_NAME or name.startswith("topopt_samples_") or name.endswith(".tmp"):
            os.remove(os.path.join(output_dir, name))


def _run_settings() -> dict[str, Any]:
    return {
        "num_samples": NUM_SAMPLES,
        "samples_per_shard": SAMPLES_PER_SHARD,
        "seed": SEED,
        "width": None if RANDOM_ASPECT else WIDTH,
        "height": None if RANDOM_ASPECT else HEIGHT,
        "nels": NELS,
        "penal": PENAL,
        "rmin": RMIN,
        "maxiter": MAXITER,
        "tolerance": TOLERANCE,
        "store_trajectory": STORE_TRAJECTORY,
    }


def generate() -> None:
    if NUM_SAMPLES < 1 or SAMPLES_PER_SHARD < 1 or BATCH_SIZE < 1 or NELS < 1:
        raise SystemExit("NUM_SAMPLES, SAMPLES_PER_SHARD, BATCH_SIZE, and NELS must be positive")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OVERWRITE:
        _clear_output(OUTPUT_DIR)

    settings = _run_settings()
    manifest = read_manifest(OUTPUT_DIR)
    if manifest is not None and manifest.get("settings") != settings:
        raise SystemExit(
            f"{os.path.join(OUTPUT_DIR, MANIFEST_NAME)} was built with different settings. "
            "Set OVERWRITE = True to replace it."
        )
    done = completed_count(manifest)
    if done > NUM_SAMPLES:
        raise SystemExit(f"output already has {done} samples, more than NUM_SAMPLES {NUM_SAMPLES}")
    if manifest is None:
        manifest = {"settings": settings, "shards": []}

    bound = trajectory_upper_bound_bytes(NUM_SAMPLES, NELS, MAXITER)
    print(
        f"target {NUM_SAMPLES} samples, {SAMPLES_PER_SHARD} per shard, {done} already written",
        flush=True,
    )
    if settings["store_trajectory"]:
        print(
            f"upper bound if every sample is {NELS}x{NELS} and uses all "
            f"{MAXITER} steps: {bound / 1024**3:.1f} GiB",
            flush=True,
        )
    if done == NUM_SAMPLES:
        print(f"nothing to do: {len(manifest['shards'])} shards in {OUTPUT_DIR}", flush=True)
        return

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        size=NUM_SAMPLES,
        seed=SEED,
    )
    if not RANDOM_ASPECT:
        loader_kwargs["width"] = WIDTH
        loader_kwargs["height"] = HEIGHT
    loader = get_problem_loader(**loader_kwargs)

    pending: list = []
    yielded = 0
    shard_index = len(manifest["shards"])
    wall_times: list[float] = []
    iter_counts: list[int] = []
    for batch in loader:
        batch_samples = list(batch.samples)
        batch_n = len(batch_samples)
        if yielded + batch_n <= done:
            yielded += batch_n
            continue
        if yielded < done:
            batch_samples = batch_samples[done - yielded :]
        yielded += batch_n

        meshed = remesh_from_coords(batch_samples, nels=NELS)
        for sample in meshed.samples:
            tic = time.perf_counter()
            optimize(
                sample,
                penal=PENAL,
                rmin=RMIN,
                maxiter=MAXITER,
                tolerance=TOLERANCE,
                store_trajectory=settings["store_trajectory"],
                plotting=False,
                quiet=True,
            )
            wall_times.append(time.perf_counter() - tic)
            if getattr(sample, "density_trajectory", None) is not None:
                iter_counts.append(int(sample.density_trajectory.shape[0]))
        _print_run_stats(done + len(wall_times), wall_times, iter_counts)
        pending.extend(meshed.samples)
        while len(pending) >= SAMPLES_PER_SHARD:
            _flush_shard(OUTPUT_DIR, manifest, shard_index, pending[:SAMPLES_PER_SHARD])
            del pending[:SAMPLES_PER_SHARD]
            shard_index += 1

    if pending:
        _flush_shard(OUTPUT_DIR, manifest, shard_index, pending)

    if wall_times:
        print(f"this run wall {_time_stats(wall_times)}", flush=True)
        if iter_counts:
            print(f"this run iters {_number_stats([float(n) for n in iter_counts])}", flush=True)

    print(
        f"wrote {completed_count(manifest)} samples across {len(manifest['shards'])} shards "
        f"in {OUTPUT_DIR}",
        flush=True,
    )


def _flush_shard(output_dir: str, manifest: dict[str, Any], shard_index: int, samples: list) -> None:
    name = shard_filename(shard_index)
    path = os.path.join(output_dir, name)
    write_shard(path, samples)
    start_index = completed_count(manifest)
    manifest["shards"].append(
        {
            "file": name,
            "num_samples": len(samples),
            "start_index": start_index,
        }
    )
    write_manifest(output_dir, manifest)
    nbytes = os.path.getsize(path)
    print(
        f"wrote {name}: {len(samples)} samples, {nbytes / 1024**2:.1f} MiB "
        f"(indices {start_index}..{start_index + len(samples) - 1})",
        flush=True,
    )


if __name__ == "__main__":
    generate()
