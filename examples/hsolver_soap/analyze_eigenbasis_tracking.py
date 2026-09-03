#!/usr/bin/env python3
"""Summarize actual SOAP eigenbasis residuals against same-step FP32 EVD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Tracking JSONL file or directory containing rank_*.jsonl",
    )
    parser.add_argument("--burn-in-step", type=int, default=20)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--tokens-per-step", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--aggregate-csv-output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load tracking-step records from one run."""
    files = sorted(path.glob("rank_*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No rank_*.jsonl files found under {path}")

    records = []
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {file_path}:{line_number}") from error
                if item.get("event") == "tracking_step":
                    records.append(item)
    if not records:
        raise ValueError(f"No tracking_step records found in {path}")
    records.sort(key=lambda item: (item["target_id"], int(item["state_step"])))
    return records


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return statistics.fmean(values)


def median(values: list[float]) -> float:
    """Return the median of a non-empty list."""
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return statistics.median(values)


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: list[float]) -> dict[str, float]:
    """Return paper-facing distribution statistics."""
    return {
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "final": values[-1],
    }


def actual_residual(record: dict[str, Any]) -> float:
    return float(record["actual_basis"]["aggregate_relative_eigen_residual"])


def evd_residual(record: dict[str, Any]) -> float:
    return float(record["evd_reference"]["aggregate_relative_eigen_residual"])


def residual_difference(record: dict[str, Any]) -> float:
    return actual_residual(record) - evd_residual(record)


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    """Return Pearson correlation, or None when either series is constant."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def validate_target_records(
    target_id: str,
    records: list[dict[str, Any]],
    expected_steps: int | None,
) -> None:
    """Reject missing, duplicate, or mixed-backend records."""
    steps = [int(record["state_step"]) for record in records]
    expected = list(range(steps[0], steps[-1] + 1))
    if steps != expected:
        raise ValueError(f"Target {target_id} has missing or duplicate state steps")
    if expected_steps is not None:
        if steps[0] != 0 or steps[-1] != expected_steps - 1 or len(steps) != expected_steps:
            raise ValueError(
                f"Target {target_id} expected {expected_steps} steps 0..{expected_steps - 1}, "
                f"got {len(steps)} steps {steps[0]}..{steps[-1]}"
            )
    backends = {record["actual_qr_backend"] for record in records}
    if len(backends) != 1:
        raise ValueError(f"Target {target_id} contains multiple actual backends: {backends}")


def target_summary(
    records: list[dict[str, Any]],
    burn_in_step: int,
) -> dict[str, Any]:
    """Summarize one parameter axis over one real optimizer trajectory."""
    late = [record for record in records if int(record["state_step"]) >= burn_in_step]
    if not late:
        raise ValueError(f"No records at or after burn-in step {burn_in_step}")

    actual = [actual_residual(record) for record in late]
    evd = [evd_residual(record) for record in late]
    difference = [residual_difference(record) for record in late]
    orthogonality = [
        float(record["actual_basis"]["normalized_orthogonality"])
        for record in late
    ]
    off_diagonal = [
        float(record["actual_basis"]["normalized_off_diagonal_energy"])
        for record in late
    ]
    factor_change_pairs = [
        (
            float(record["factor_change"]["relative_frobenius_norm"]),
            actual_residual(record),
        )
        for record in late
        if record["factor_change"]["available"]
    ]
    factor_changes = [pair[0] for pair in factor_change_pairs]
    residuals_with_change = [pair[1] for pair in factor_change_pairs]

    return {
        "target_id": records[0]["target_id"],
        "rank": records[0]["rank"],
        "pipeline_parallel_rank": records[0]["pipeline_parallel_rank"],
        "tensor_parallel_rank": records[0]["tensor_parallel_rank"],
        "parameter_name": records[0]["parameter_name"],
        "parameter_shape": records[0]["parameter_shape"],
        "axis_index": records[0]["axis_index"],
        "dimension": records[0]["dimension"],
        "actual_qr_backend": records[0]["actual_qr_backend"],
        "records": len(records),
        "late_records": len(late),
        "state_step_first": int(records[0]["state_step"]),
        "state_step_last": int(records[-1]["state_step"]),
        "actual_feature_residual": numeric_summary(actual),
        "same_step_evd_feature_residual": numeric_summary(evd),
        "actual_minus_evd_feature_residual": numeric_summary(difference),
        "actual_normalized_orthogonality": numeric_summary(orthogonality),
        "actual_normalized_off_diagonal_energy": numeric_summary(off_diagonal),
        "factor_relative_change": (
            numeric_summary(factor_changes) if factor_changes else None
        ),
        "factor_change_to_actual_residual_correlation": pearson_correlation(
            factor_changes,
            residuals_with_change,
        ),
        "actual_all_finite": all(
            bool(record["actual_basis"]["all_finite"]) for record in records
        ),
        "evd_all_finite": all(
            bool(record["evd_reference"]["all_finite"]) for record in records
        ),
    }


def csv_row(
    record: dict[str, Any],
    expected_steps: int | None,
    tokens_per_step: int | None,
) -> dict[str, Any]:
    optimizer_step = int(record["optimizer_step"])
    return {
        "experiment": record["experiment"],
        "actual_qr_backend": record["actual_qr_backend"],
        "target_id": record["target_id"],
        "rank": record["rank"],
        "pipeline_parallel_rank": record["pipeline_parallel_rank"],
        "tensor_parallel_rank": record["tensor_parallel_rank"],
        "parameter_name": record["parameter_name"],
        "parameter_shape": "x".join(str(value) for value in record["parameter_shape"]),
        "axis_index": record["axis_index"],
        "dimension": record["dimension"],
        "state_step": record["state_step"],
        "optimizer_step": optimizer_step,
        "normalized_progress": (
            optimizer_step / expected_steps if expected_steps is not None else ""
        ),
        "cumulative_tokens": (
            optimizer_step * tokens_per_step if tokens_per_step is not None else ""
        ),
        "basis_update_method": record["basis_update_method"],
        "actual_feature_residual": actual_residual(record),
        "same_step_evd_feature_residual": evd_residual(record),
        "actual_minus_evd_feature_residual": residual_difference(record),
        "actual_normalized_orthogonality": record["actual_basis"][
            "normalized_orthogonality"
        ],
        "actual_normalized_off_diagonal_energy": record["actual_basis"][
            "normalized_off_diagonal_energy"
        ],
        "factor_frobenius_norm": record["actual_basis"]["factor_frobenius_norm"],
        "factor_relative_change": (
            record["factor_change"]["relative_frobenius_norm"]
            if record["factor_change"]["available"]
            else ""
        ),
        "actual_all_finite": record["actual_basis"]["all_finite"],
        "evd_all_finite": record["evd_reference"]["all_finite"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate all tracked targets into one median curve per optimizer step."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["state_step"])].append(row)

    result = []
    for state_step in sorted(by_step):
        step_rows = by_step[state_step]
        result.append(
            {
                "experiment": step_rows[0]["experiment"],
                "actual_qr_backend": step_rows[0]["actual_qr_backend"],
                "state_step": state_step,
                "optimizer_step": step_rows[0]["optimizer_step"],
                "normalized_progress": step_rows[0]["normalized_progress"],
                "cumulative_tokens": step_rows[0]["cumulative_tokens"],
                "target_count": len(step_rows),
                "median_actual_feature_residual": median(
                    [float(row["actual_feature_residual"]) for row in step_rows]
                ),
                "median_same_step_evd_feature_residual": median(
                    [float(row["same_step_evd_feature_residual"]) for row in step_rows]
                ),
                "median_actual_minus_evd_feature_residual": median(
                    [float(row["actual_minus_evd_feature_residual"]) for row in step_rows]
                ),
                "median_actual_normalized_orthogonality": median(
                    [float(row["actual_normalized_orthogonality"]) for row in step_rows]
                ),
                "median_actual_normalized_off_diagonal_energy": median(
                    [float(row["actual_normalized_off_diagonal_energy"]) for row in step_rows]
                ),
            }
        )
    return result


def main() -> None:
    """Load one real training run and emit per-target and median-curve outputs."""
    args = parse_args()
    records = load_records(args.input)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["target_id"]].append(record)

    summaries = {}
    for target_id, target_records in sorted(grouped.items()):
        validate_target_records(target_id, target_records, args.expected_steps)
        summaries[target_id] = target_summary(target_records, args.burn_in_step)

    experiments = {record["experiment"] for record in records}
    backends = {record["actual_qr_backend"] for record in records}
    if len(experiments) != 1 or len(backends) != 1:
        raise ValueError(
            f"One analysis input must contain one experiment/backend, got {experiments}, {backends}"
        )

    summary = {
        "input": str(args.input.resolve()),
        "experiment": next(iter(experiments)),
        "actual_qr_backend": next(iter(backends)),
        "burn_in_step": args.burn_in_step,
        "expected_steps": args.expected_steps,
        "tokens_per_step": args.tokens_per_step,
        "target_count": len(summaries),
        "targets": summaries,
    }

    print("=" * 72)
    print("ACTUAL SOAP BASIS VERSUS SAME-STEP FP32 EVD")
    print("=" * 72)
    print(f"experiment: {summary['experiment']}")
    print(f"actual_qr_backend: {summary['actual_qr_backend']}")
    print(f"target_count: {summary['target_count']}")
    for target_id, target in summaries.items():
        print("-" * 72)
        print(target_id)
        print(f"parameter: {target['parameter_name']}")
        print(f"records: {target['records']}")
        print(
            "actual residual late median: "
            f"{target['actual_feature_residual']['median']}"
        )
        print(
            "same-step EVD residual late median: "
            f"{target['same_step_evd_feature_residual']['median']}"
        )
        print(
            "actual-EVD residual late median: "
            f"{target['actual_minus_evd_feature_residual']['median']}"
        )
        print(f"actual all finite: {target['actual_all_finite']}")
        print(f"EVD all finite: {target['evd_all_finite']}")

    rows = [
        csv_row(record, args.expected_steps, args.tokens_per_step)
        for record in records
    ]
    if args.csv_output is not None:
        write_csv(args.csv_output, rows)
        print(f"Per-target per-step CSV: {args.csv_output}")
    if args.aggregate_csv_output is not None:
        write_csv(args.aggregate_csv_output, aggregate_rows(rows))
        print(f"Per-step target-median CSV: {args.aggregate_csv_output}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON summary: {args.json_output}")


if __name__ == "__main__":
    main()
