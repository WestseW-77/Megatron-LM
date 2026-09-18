#!/usr/bin/env python3
"""Aggregate step, optimizer, and SOAP QR timings for one 30-step run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ITERATIONS = tuple(range(20, 31))
EXPECTED_WORLD_SIZE = 4
ACTUAL_BACKENDS = ("torch", "hsolver", "torch_fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_timing_rows(run_directory: Path) -> list[dict[str, int | float]]:
    timing_directory = run_directory / "timing"
    rank_files = sorted(timing_directory.glob("timing_rank*.csv"))
    if len(rank_files) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"expected {EXPECTED_WORLD_SIZE} optimizer timing files in {timing_directory}, "
            f"found {len(rank_files)}"
        )

    rows: list[dict[str, int | float]] = []
    for rank_file in rank_files:
        parsed = _read_csv(rank_file)
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


def read_qr_rows(run_directory: Path) -> list[dict[str, int | float | str]]:
    timing_directory = run_directory / "timing"
    rank_files = sorted(timing_directory.glob("qr_timing_rank*.csv"))
    if len(rank_files) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"expected {EXPECTED_WORLD_SIZE} QR timing files in {timing_directory}, "
            f"found {len(rank_files)}"
        )

    rows: list[dict[str, int | float | str]] = []
    for rank_file in rank_files:
        parsed = _read_csv(rank_file)
        if not parsed:
            raise RuntimeError(f"{rank_file} has no QR timing rows")
        for row in parsed:
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "rank": int(row["rank"]),
                    "world_size": int(row["world_size"]),
                    "cuda_device": int(row["cuda_device"]),
                    "global_batch_size": int(row["global_batch_size"]),
                    "call_index": int(row["call_index"]),
                    "requested_backend": row["requested_backend"],
                    "actual_backend": row["actual_backend"],
                    "rows": int(row["rows"]),
                    "columns": int(row["columns"]),
                    "dtype": row["dtype"],
                    "qr_ms": float(row["qr_ms"]),
                }
            )
    return rows


def validate(
    timing_rows: list[dict[str, int | float]],
    qr_rows: list[dict[str, int | float | str]],
) -> tuple[int, str]:
    all_rows = timing_rows + qr_rows
    world_sizes = {int(row["world_size"]) for row in all_rows}
    if world_sizes != {EXPECTED_WORLD_SIZE}:
        raise RuntimeError(f"unexpected world sizes: {sorted(world_sizes)}")
    ranks = {int(row["rank"]) for row in all_rows}
    if ranks != set(range(EXPECTED_WORLD_SIZE)):
        raise RuntimeError(f"unexpected ranks: {sorted(ranks)}")
    batch_sizes = {int(row["global_batch_size"]) for row in all_rows}
    if len(batch_sizes) != 1:
        raise RuntimeError(f"timing files disagree on GBS: {sorted(batch_sizes)}")

    expected_keys = {
        (iteration, rank)
        for iteration in EXPECTED_ITERATIONS
        for rank in range(EXPECTED_WORLD_SIZE)
    }
    timing_keys = {(int(row["iteration"]), int(row["rank"])) for row in timing_rows}
    qr_keys = {(int(row["iteration"]), int(row["rank"])) for row in qr_rows}
    if timing_keys != expected_keys:
        raise RuntimeError("optimizer timing rows do not cover every expected iteration and rank")
    if qr_keys != expected_keys:
        raise RuntimeError("QR timing rows do not cover every expected iteration and rank")

    requested_backends = {str(row["requested_backend"]) for row in qr_rows}
    if len(requested_backends) != 1:
        raise RuntimeError(f"QR rows disagree on requested backend: {sorted(requested_backends)}")
    requested_backend = requested_backends.pop()
    if requested_backend not in ("torch", "hsolver"):
        raise RuntimeError(f"unexpected requested QR backend: {requested_backend}")
    actual_backends = {str(row["actual_backend"]) for row in qr_rows}
    unexpected_actual = actual_backends.difference(ACTUAL_BACKENDS)
    if unexpected_actual:
        raise RuntimeError(f"unexpected actual QR backends: {sorted(unexpected_actual)}")
    if requested_backend == "torch" and actual_backends != {"torch"}:
        raise RuntimeError(f"Torch run has unexpected actual backends: {sorted(actual_backends)}")

    return batch_sizes.pop(), requested_backend


def _statistics(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate(
    timing_rows: list[dict[str, int | float]],
    qr_rows: list[dict[str, int | float | str]],
    global_batch_size: int,
    requested_backend: str,
):
    qr_by_rank_iteration: dict[tuple[int, int], list[dict[str, int | float | str]]] = defaultdict(list)
    for row in qr_rows:
        qr_by_rank_iteration[(int(row["iteration"]), int(row["rank"]))].append(row)

    per_rank_iteration = []
    for timing_row in timing_rows:
        iteration = int(timing_row["iteration"])
        rank = int(timing_row["rank"])
        calls = qr_by_rank_iteration[(iteration, rank)]
        backend_ms = {
            backend: sum(float(call["qr_ms"]) for call in calls if call["actual_backend"] == backend)
            for backend in ACTUAL_BACKENDS
        }
        qr_ms = sum(backend_ms.values())
        optimizer_ms = float(timing_row["optimizer_ms"])
        step_ms = float(timing_row["step_ms"])
        per_rank_iteration.append(
            {
                "iteration": iteration,
                "rank": rank,
                "step_ms": step_ms,
                "optimizer_ms": optimizer_ms,
                "qr_ms": qr_ms,
                "non_qr_optimizer_ms": optimizer_ms - qr_ms,
                "qr_call_count": len(calls),
                "torch_qr_ms": backend_ms["torch"],
                "hsolver_qr_ms": backend_ms["hsolver"],
                "torch_fallback_qr_ms": backend_ms["torch_fallback"],
                "qr_optimizer_fraction": qr_ms / optimizer_ms,
                "qr_step_fraction": qr_ms / step_ms,
            }
        )

    grouped: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for row in per_rank_iteration:
        grouped[int(row["iteration"])].append(row)

    per_iteration = []
    for iteration in EXPECTED_ITERATIONS:
        iteration_rows = grouped[iteration]
        step_row = max(iteration_rows, key=lambda row: float(row["step_ms"]))
        optimizer_row = max(iteration_rows, key=lambda row: float(row["optimizer_ms"]))
        qr_row = max(iteration_rows, key=lambda row: float(row["qr_ms"]))
        step_ms = float(step_row["step_ms"])
        optimizer_ms = float(optimizer_row["optimizer_ms"])
        critical_qr_ms = float(optimizer_row["qr_ms"])
        per_iteration.append(
            {
                "iteration": iteration,
                "step_max_ms": step_ms,
                "step_max_rank": int(step_row["rank"]),
                "optimizer_max_ms": optimizer_ms,
                "optimizer_max_rank": int(optimizer_row["rank"]),
                "qr_ms_on_optimizer_max_rank": critical_qr_ms,
                "non_qr_optimizer_ms_on_optimizer_max_rank": optimizer_ms - critical_qr_ms,
                "qr_max_ms": float(qr_row["qr_ms"]),
                "qr_max_rank": int(qr_row["rank"]),
                "optimizer_fraction": optimizer_ms / step_ms,
                "qr_optimizer_fraction_on_optimizer_max_rank": critical_qr_ms / optimizer_ms,
                "qr_step_fraction_on_optimizer_max_rank": critical_qr_ms / step_ms,
            }
        )

    step_times = [float(row["step_max_ms"]) for row in per_iteration]
    optimizer_times = [float(row["optimizer_max_ms"]) for row in per_iteration]
    critical_qr_times = [float(row["qr_ms_on_optimizer_max_rank"]) for row in per_iteration]
    non_qr_times = [float(row["non_qr_optimizer_ms_on_optimizer_max_rank"]) for row in per_iteration]
    qr_max_times = [float(row["qr_max_ms"]) for row in per_iteration]
    call_counts = [int(row["qr_call_count"]) for row in per_rank_iteration]

    actual_backend_totals = {}
    for backend in ACTUAL_BACKENDS:
        backend_rows = [row for row in qr_rows if row["actual_backend"] == backend]
        actual_backend_totals[backend] = {
            "call_count": len(backend_rows),
            "summed_gpu_work_ms_all_ranks": sum(float(row["qr_ms"]) for row in backend_rows),
        }

    summary = {
        "requested_qr_backend": requested_backend,
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": global_batch_size,
        "tokens_per_optimizer_step": global_batch_size * 8192,
        "measurement_start_iteration": EXPECTED_ITERATIONS[0],
        "measurement_end_iteration": EXPECTED_ITERATIONS[-1],
        "measurement_count": len(EXPECTED_ITERATIONS),
        "rank_aggregation": (
            "step and optimizer use per-iteration maxima across ranks; QR/non-QR values use "
            "the same rank that had the maximum optimizer duration"
        ),
        "step_ms": _statistics(step_times),
        "optimizer_ms": _statistics(optimizer_times),
        "qr_ms_on_optimizer_critical_rank": _statistics(critical_qr_times),
        "qr_max_ms": _statistics(qr_max_times),
        "non_qr_optimizer_ms_on_optimizer_critical_rank": _statistics(non_qr_times),
        "optimizer_fraction_ratio_of_sums": sum(optimizer_times) / sum(step_times),
        "qr_optimizer_fraction_ratio_of_sums": sum(critical_qr_times) / sum(optimizer_times),
        "qr_step_fraction_ratio_of_sums": sum(critical_qr_times) / sum(step_times),
        "qr_call_count_per_rank_iteration": {
            "minimum": min(call_counts),
            "maximum": max(call_counts),
            "counts": dict(sorted(Counter(call_counts).items())),
        },
        "actual_backend_totals": actual_backend_totals,
    }
    return per_rank_iteration, per_iteration, summary


def dimension_summary(qr_rows: list[dict[str, int | float | str]]):
    grouped: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in qr_rows:
        key = (
            str(row["requested_backend"]),
            str(row["actual_backend"]),
            int(row["rows"]),
            int(row["columns"]),
            str(row["dtype"]),
        )
        grouped[key].append(float(row["qr_ms"]))

    output = []
    for key, values in sorted(grouped.items()):
        requested_backend, actual_backend, rows, columns, dtype = key
        output.append(
            {
                "requested_backend": requested_backend,
                "actual_backend": actual_backend,
                "rows": rows,
                "columns": columns,
                "dtype": dtype,
                "call_count": len(values),
                "mean_call_ms": statistics.fmean(values),
                "median_call_ms": statistics.median(values),
                "minimum_call_ms": min(values),
                "maximum_call_ms": max(values),
                "summed_gpu_work_ms_all_ranks": sum(values),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_results(run_directory: Path, per_rank_iteration, per_iteration, dimensions, summary) -> None:
    _write_csv(run_directory / "qr_timing_per_rank_iteration.csv", per_rank_iteration)
    _write_csv(run_directory / "qr_timing_per_iteration.csv", per_iteration)
    _write_csv(run_directory / "qr_timing_by_dimension.csv", dimensions)
    summary_path = run_directory / "qr_timing_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)


def main() -> None:
    args = parse_args()
    timing_rows = read_timing_rows(args.run_directory)
    qr_rows = read_qr_rows(args.run_directory)
    global_batch_size, requested_backend = validate(timing_rows, qr_rows)
    per_rank_iteration, per_iteration, summary = aggregate(
        timing_rows,
        qr_rows,
        global_batch_size,
        requested_backend,
    )
    dimensions = dimension_summary(qr_rows)
    write_results(args.run_directory, per_rank_iteration, per_iteration, dimensions, summary)

    print(f"BACKEND={requested_backend}")
    print(f"GBS={global_batch_size}")
    print(f"MEASURED_ITERATIONS={EXPECTED_ITERATIONS[0]}..{EXPECTED_ITERATIONS[-1]}")
    print(f"STEP_MEAN_MS={summary['step_ms']['mean']:.3f}")
    print(f"OPTIMIZER_MEAN_MS={summary['optimizer_ms']['mean']:.3f}")
    print(
        "QR_MEAN_MS_ON_OPTIMIZER_CRITICAL_RANK="
        f"{summary['qr_ms_on_optimizer_critical_rank']['mean']:.3f}"
    )
    print(
        "NON_QR_OPTIMIZER_MEAN_MS_ON_OPTIMIZER_CRITICAL_RANK="
        f"{summary['non_qr_optimizer_ms_on_optimizer_critical_rank']['mean']:.3f}"
    )
    print(
        "QR_OPTIMIZER_FRACTION="
        f"{100.0 * summary['qr_optimizer_fraction_ratio_of_sums']:.4f}%"
    )
    print(f"QR_STEP_FRACTION={100.0 * summary['qr_step_fraction_ratio_of_sums']:.4f}%")


if __name__ == "__main__":
    main()
