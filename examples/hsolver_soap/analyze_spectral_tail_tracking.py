#!/usr/bin/env python3
"""Summarize one independent SOAP trajectory against its own FP32 EVD references."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Tracking directory containing rank_*.jsonl files",
    )
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--burn-in-state-step", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "final": values[-1],
    }


def optional_numeric_summary(values: list[float | None]) -> dict[str, float] | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    return numeric_summary(finite) if finite else None


def load_records(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("rank_*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No rank_*.jsonl files found under {path}")
    records = []
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {file_path}:{line_number}"
                    ) from error
                if record.get("event") == "tracking_step":
                    records.append(record)
    if not records:
        raise ValueError(f"No tracking_step records found under {path}")
    records.sort(key=lambda item: (item["target_id"], int(item["state_step"])))
    return records


def validate_target(
    target_id: str,
    records: list[dict[str, Any]],
    expected_steps: int,
) -> str:
    steps = [int(record["state_step"]) for record in records]
    if steps != list(range(expected_steps)):
        raise ValueError(
            f"Target {target_id} expected state steps 0..{expected_steps - 1}, "
            f"got {steps}"
        )
    backends = {record["actual_qr_backend"] for record in records}
    if len(backends) != 1:
        raise ValueError(f"Target {target_id} contains multiple backends: {backends}")
    backend = next(iter(backends))
    initial = records[0]["spectral_localization"]
    if (
        initial.get("schema_version") != 3
        or initial.get("mode") != "actual_only"
        or not initial["enabled"]
        or initial["available"]
    ):
        raise ValueError(f"Target {target_id} has an invalid EVD-initialization record")
    for record in records[1:]:
        spectral = record["spectral_localization"]
        if (
            spectral.get("schema_version") != 3
            or spectral.get("mode") != "actual_only"
            or not spectral["available"]
        ):
            raise ValueError(
                f"Target {target_id} state {record['state_step']} is not actual-only"
            )
        if spectral["actual_training_backend"] != backend:
            raise ValueError(
                f"Target {target_id} state {record['state_step']} backend mismatch"
            )
        if not spectral["actual"]["all_finite"]:
            raise ValueError(
                f"Target {target_id} state {record['state_step']} is nonfinite"
            )
    return backend


def per_threshold_row(record: dict[str, Any], threshold: str) -> dict[str, Any]:
    spectral = record["spectral_localization"]
    actual = spectral["actual"]
    threshold_metrics = actual["thresholds"][threshold]
    return {
        "experiment": record["experiment"],
        "actual_qr_backend": record["actual_qr_backend"],
        "target_id": record["target_id"],
        "rank": record["rank"],
        "parameter_name": record["parameter_name"],
        "axis_index": record["axis_index"],
        "dimension": record["dimension"],
        "state_step": record["state_step"],
        "optimizer_step": record["optimizer_step"],
        "trace_energy_threshold": float(threshold),
        "head_dimension": threshold_metrics["head_dimension"],
        "head_dimension_fraction": threshold_metrics["head_dimension_fraction"],
        "captured_trace_energy_fraction": threshold_metrics[
            "captured_trace_energy_fraction"
        ],
        "boundary_relative_eigengap": threshold_metrics[
            "boundary_relative_eigengap"
        ],
        "effective_rank": actual["effective_rank"],
        "aggregate_relative_eigen_residual": actual[
            "aggregate_relative_eigen_residual"
        ],
        "leading_subspace_overlap": threshold_metrics[
            "leading_subspace_overlap"
        ],
        "tail_residual_energy_share": threshold_metrics[
            "tail_residual_energy_share"
        ],
        "head_normalized_residual_energy": threshold_metrics[
            "head_normalized_residual_energy"
        ],
        "tail_normalized_residual_energy": threshold_metrics[
            "tail_normalized_residual_energy"
        ],
        "rayleigh_to_relative_residual_spearman": actual[
            "rayleigh_to_relative_residual_spearman"
        ],
    }


def summarize_target(
    records: list[dict[str, Any]],
    burn_in_state_step: int,
) -> dict[str, Any]:
    late = [
        record
        for record in records
        if int(record["state_step"]) >= burn_in_state_step
        and record["spectral_localization"]["available"]
    ]
    if not late:
        raise ValueError(
            f"No spectral records at or after state step {burn_in_state_step}"
        )
    actual_metrics = [record["spectral_localization"]["actual"] for record in late]
    thresholds = tuple(actual_metrics[0]["thresholds"])
    result: dict[str, Any] = {
        "target_id": records[0]["target_id"],
        "rank": records[0]["rank"],
        "parameter_name": records[0]["parameter_name"],
        "axis_index": records[0]["axis_index"],
        "dimension": records[0]["dimension"],
        "actual_qr_backend": records[0]["actual_qr_backend"],
        "records": len(records),
        "spectral_records_summarized": len(late),
        "aggregate_relative_eigen_residual": numeric_summary(
            [
                float(metrics["aggregate_relative_eigen_residual"])
                for metrics in actual_metrics
            ]
        ),
        "effective_rank": numeric_summary(
            [float(metrics["effective_rank"]) for metrics in actual_metrics]
        ),
        "rayleigh_to_relative_residual_spearman": optional_numeric_summary(
            [
                metrics["rayleigh_to_relative_residual_spearman"]
                for metrics in actual_metrics
            ]
        ),
        "thresholds": {},
    }
    for threshold in thresholds:
        metrics = [item["thresholds"][threshold] for item in actual_metrics]
        result["thresholds"][threshold] = {
            "head_dimension": numeric_summary(
                [float(item["head_dimension"]) for item in metrics]
            ),
            "head_dimension_fraction": numeric_summary(
                [float(item["head_dimension_fraction"]) for item in metrics]
            ),
            "boundary_relative_eigengap": optional_numeric_summary(
                [item["boundary_relative_eigengap"] for item in metrics]
            ),
            "leading_subspace_overlap": numeric_summary(
                [
                    float(item["leading_subspace_overlap"])
                    for item in metrics
                ]
            ),
            "tail_residual_energy_share": numeric_summary(
                [float(item["tail_residual_energy_share"]) for item in metrics]
            ),
            "head_normalized_residual_energy": numeric_summary(
                [
                    float(item["head_normalized_residual_energy"])
                    for item in metrics
                ]
            ),
            "tail_normalized_residual_energy": numeric_summary(
                [
                    float(item["tail_normalized_residual_energy"])
                    for item in metrics
                ]
            ),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["target_id"]].append(record)

    summaries = {}
    rows = []
    run_backends = set()
    for target_id, target_records in sorted(grouped.items()):
        run_backends.add(
            validate_target(target_id, target_records, args.expected_steps)
        )
        summaries[target_id] = summarize_target(
            target_records,
            args.burn_in_state_step,
        )
        for record in target_records[1:]:
            actual = record["spectral_localization"]["actual"]
            for threshold in actual["thresholds"]:
                rows.append(per_threshold_row(record, threshold))

    experiments = {record["experiment"] for record in records}
    if len(experiments) != 1 or len(run_backends) != 1:
        raise ValueError(
            f"Expected one experiment/backend, got {experiments}, {run_backends}"
        )
    backend = next(iter(run_backends))
    summary = {
        "schema_version": 3,
        "comparison_mode": "actual_only_independent_trajectory",
        "input": str(args.input.resolve()),
        "experiment": next(iter(experiments)),
        "actual_qr_backend": backend,
        "expected_steps": args.expected_steps,
        "burn_in_state_step": args.burn_in_state_step,
        "target_count": len(summaries),
        "reference": "fresh FP32 EVD of this trajectory's own current factor",
        "targets": summaries,
    }

    print("=" * 76)
    print("INDEPENDENT ACTUAL SOAP TRAJECTORY VERSUS OWN FP32 EVD")
    print("=" * 76)
    print(f"experiment: {summary['experiment']}")
    print(f"actual backend: {backend}")
    print(f"targets: {summary['target_count']}")
    print(f"steps per target: {args.expected_steps} (state 0 is EVD initialization)")
    for target_id, target in summaries.items():
        print("-" * 76)
        print(target_id)
        print(
            "aggregate residual median: "
            f"{target['aggregate_relative_eigen_residual']['median']}"
        )
        print(f"effective rank median: {target['effective_rank']['median']}")
        for threshold in ("0.9", "0.99", "0.999"):
            if threshold not in target["thresholds"]:
                continue
            metrics = target["thresholds"][threshold]
            print(
                f"trace head {threshold}: median dimension="
                f"{metrics['head_dimension']['median']}, "
                "leading subspace overlap="
                f"{metrics['leading_subspace_overlap']['median']}, "
                f"tail residual share="
                f"{metrics['tail_residual_energy_share']['median']}"
            )

    if args.csv_output is not None:
        write_csv(args.csv_output, rows)
        print(f"Per-target per-step spectral CSV: {args.csv_output}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Spectral JSON summary: {args.json_output}")


if __name__ == "__main__":
    main()
