#!/usr/bin/env python3
"""Aggregate rank-local optimizer fraction timing for one 30-step run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_ITERATIONS = tuple(range(20, 31))
EXPECTED_WORLD_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    return parser.parse_args()


def read_rank_files(run_directory: Path) -> list[dict[str, int | float]]:
    timing_directory = run_directory / "timing"
    rank_files = sorted(timing_directory.glob("timing_rank*.csv"))
    if len(rank_files) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"expected {EXPECTED_WORLD_SIZE} rank files in {timing_directory}, "
            f"found {len(rank_files)}"
        )

    rows: list[dict[str, int | float]] = []
    for rank_file in rank_files:
        with rank_file.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        if len(parsed) != len(EXPECTED_ITERATIONS):
            raise RuntimeError(
                f"{rank_file} has {len(parsed)} rows; expected {len(EXPECTED_ITERATIONS)}"
            )
        for row in parsed:
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "rank": int(row["rank"]),
                    "world_size": int(row["world_size"]),
                    "cuda_device": int(row["cuda_device"]),
                    "global_batch_size": int(row["global_batch_size"]),
                    "step_ms": float(row["step_ms"]),
                    "optimizer_ms": float(row["optimizer_ms"]),
                }
            )
    return rows


def validate(rows: list[dict[str, int | float]]) -> int:
    world_sizes = {int(row["world_size"]) for row in rows}
    if world_sizes != {EXPECTED_WORLD_SIZE}:
        raise RuntimeError(f"unexpected world sizes: {sorted(world_sizes)}")
    ranks = {int(row["rank"]) for row in rows}
    if ranks != set(range(EXPECTED_WORLD_SIZE)):
        raise RuntimeError(f"unexpected ranks: {sorted(ranks)}")
    batch_sizes = {int(row["global_batch_size"]) for row in rows}
    if len(batch_sizes) != 1:
        raise RuntimeError(f"rank files disagree on GBS: {sorted(batch_sizes)}")

    by_rank: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_rank[int(row["rank"])].add(int(row["iteration"]))
    expected = set(EXPECTED_ITERATIONS)
    for rank, iterations in by_rank.items():
        if iterations != expected:
            raise RuntimeError(
                f"rank {rank} iterations differ from {EXPECTED_ITERATIONS}: {sorted(iterations)}"
            )
    return batch_sizes.pop()


def aggregate(rows: list[dict[str, int | float]], global_batch_size: int):
    grouped: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["iteration"])].append(row)

    per_iteration = []
    for iteration in EXPECTED_ITERATIONS:
        iteration_rows = grouped[iteration]
        step_row = max(iteration_rows, key=lambda row: float(row["step_ms"]))
        optimizer_row = max(iteration_rows, key=lambda row: float(row["optimizer_ms"]))
        step_ms = float(step_row["step_ms"])
        optimizer_ms = float(optimizer_row["optimizer_ms"])
        per_iteration.append(
            {
                "iteration": iteration,
                "step_max_ms": step_ms,
                "step_max_rank": int(step_row["rank"]),
                "optimizer_max_ms": optimizer_ms,
                "optimizer_max_rank": int(optimizer_row["rank"]),
                "optimizer_fraction": optimizer_ms / step_ms,
            }
        )

    step_times = [row["step_max_ms"] for row in per_iteration]
    optimizer_times = [row["optimizer_max_ms"] for row in per_iteration]
    fractions = [row["optimizer_fraction"] for row in per_iteration]
    summary = {
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": global_batch_size,
        "tokens_per_optimizer_step": global_batch_size * 8192,
        "measurement_start_iteration": EXPECTED_ITERATIONS[0],
        "measurement_end_iteration": EXPECTED_ITERATIONS[-1],
        "measurement_count": len(EXPECTED_ITERATIONS),
        "rank_aggregation": "maximum duration across ranks for each iteration",
        "step_ms": {
            "mean": statistics.fmean(step_times),
            "median": statistics.median(step_times),
            "stdev": statistics.stdev(step_times),
            "minimum": min(step_times),
            "maximum": max(step_times),
        },
        "optimizer_ms": {
            "mean": statistics.fmean(optimizer_times),
            "median": statistics.median(optimizer_times),
            "stdev": statistics.stdev(optimizer_times),
            "minimum": min(optimizer_times),
            "maximum": max(optimizer_times),
        },
        "optimizer_fraction_ratio_of_sums": sum(optimizer_times) / sum(step_times),
        "optimizer_fraction_mean_of_iterations": statistics.fmean(fractions),
        "optimizer_fraction_median_of_iterations": statistics.median(fractions),
    }
    return per_iteration, summary


def write_results(run_directory: Path, per_iteration, summary) -> None:
    per_iteration_path = run_directory / "timing_per_iteration.csv"
    with per_iteration_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_iteration[0].keys())
        writer.writeheader()
        writer.writerows(per_iteration)

    summary_path = run_directory / "timing_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)


def main() -> None:
    args = parse_args()
    rows = read_rank_files(args.run_directory)
    global_batch_size = validate(rows)
    per_iteration, summary = aggregate(rows, global_batch_size)
    write_results(args.run_directory, per_iteration, summary)
    print(f"GBS={global_batch_size}")
    print(f"MEASURED_ITERATIONS={EXPECTED_ITERATIONS[0]}..{EXPECTED_ITERATIONS[-1]}")
    print(f"STEP_MEAN_MS={summary['step_ms']['mean']:.3f}")
    print(f"OPTIMIZER_MEAN_MS={summary['optimizer_ms']['mean']:.3f}")
    print(
        "OPTIMIZER_FRACTION="
        f"{100.0 * summary['optimizer_fraction_ratio_of_sums']:.4f}%"
    )


if __name__ == "__main__":
    main()
