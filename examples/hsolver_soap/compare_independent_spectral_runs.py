#!/usr/bin/env python3
"""Compare summaries from independent Torch and Hsolver SOAP trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != 3
        or summary.get("comparison_mode") != "actual_only_independent_trajectory"
    ):
        raise ValueError(f"{path} is not an actual-only spectral summary")
    return summary


def assign_backends(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_backend = {
        first["actual_qr_backend"]: first,
        second["actual_qr_backend"]: second,
    }
    if set(by_backend) != {"torch", "hsolver"}:
        raise ValueError(
            "Comparison requires one independent Torch summary and one Hsolver summary"
        )
    return by_backend["torch"], by_backend["hsolver"]


def validate_compatible(
    torch_summary: dict[str, Any],
    hsolver_summary: dict[str, Any],
) -> None:
    for field in ("expected_steps", "burn_in_state_step", "target_count"):
        if torch_summary[field] != hsolver_summary[field]:
            raise ValueError(f"Run summaries disagree on {field}")
    torch_targets = torch_summary["targets"]
    hsolver_targets = hsolver_summary["targets"]
    if set(torch_targets) != set(hsolver_targets):
        raise ValueError("Run summaries do not contain the same target IDs")
    for target_id in torch_targets:
        torch_target = torch_targets[target_id]
        hsolver_target = hsolver_targets[target_id]
        for field in ("rank", "axis_index", "dimension"):
            if torch_target[field] != hsolver_target[field]:
                raise ValueError(f"Target {target_id} disagrees on {field}")
        if set(torch_target["thresholds"]) != set(hsolver_target["thresholds"]):
            raise ValueError(f"Target {target_id} has different energy thresholds")


def median(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]["median"])


def comparison_rows(
    torch_summary: dict[str, Any],
    hsolver_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for target_id, torch_target in sorted(torch_summary["targets"].items()):
        hsolver_target = hsolver_summary["targets"][target_id]
        torch_residual = median(torch_target, "aggregate_relative_eigen_residual")
        hsolver_residual = median(
            hsolver_target,
            "aggregate_relative_eigen_residual",
        )
        for threshold, torch_metrics in torch_target["thresholds"].items():
            hsolver_metrics = hsolver_target["thresholds"][threshold]
            torch_overlap = median(torch_metrics, "leading_subspace_overlap")
            hsolver_overlap = median(
                hsolver_metrics,
                "leading_subspace_overlap",
            )
            torch_tail = median(torch_metrics, "tail_residual_energy_share")
            hsolver_tail = median(hsolver_metrics, "tail_residual_energy_share")
            rows.append(
                {
                    "target_id": target_id,
                    "rank": torch_target["rank"],
                    "axis_index": torch_target["axis_index"],
                    "dimension": torch_target["dimension"],
                    "trace_energy_threshold": float(threshold),
                    "torch_aggregate_residual_median": torch_residual,
                    "hsolver_aggregate_residual_median": hsolver_residual,
                    "descriptive_hsolver_to_torch_residual_ratio": (
                        hsolver_residual / torch_residual
                        if torch_residual != 0.0
                        else None
                    ),
                    "torch_effective_rank_median": median(
                        torch_target,
                        "effective_rank",
                    ),
                    "hsolver_effective_rank_median": median(
                        hsolver_target,
                        "effective_rank",
                    ),
                    "torch_head_dimension_median": median(
                        torch_metrics,
                        "head_dimension",
                    ),
                    "hsolver_head_dimension_median": median(
                        hsolver_metrics,
                        "head_dimension",
                    ),
                    "torch_leading_subspace_overlap_median": torch_overlap,
                    "hsolver_leading_subspace_overlap_median": hsolver_overlap,
                    "descriptive_hsolver_minus_torch_leading_subspace_overlap": (
                        hsolver_overlap - torch_overlap
                    ),
                    "torch_tail_residual_share_median": torch_tail,
                    "hsolver_tail_residual_share_median": hsolver_tail,
                    "descriptive_hsolver_minus_torch_tail_share": (
                        hsolver_tail - torch_tail
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    first = load_summary(args.first)
    second = load_summary(args.second)
    torch_summary, hsolver_summary = assign_backends(first, second)
    validate_compatible(torch_summary, hsolver_summary)
    rows = comparison_rows(torch_summary, hsolver_summary)

    result = {
        "schema_version": 2,
        "comparison_mode": "descriptive_comparison_of_independent_trajectories",
        "warning": (
            "Torch and Hsolver evolve different K_t and Q_t states; differences are "
            "end-to-end descriptive comparisons, not same-input solver error."
        ),
        "torch_experiment": torch_summary["experiment"],
        "hsolver_experiment": hsolver_summary["experiment"],
        "expected_steps": torch_summary["expected_steps"],
        "rows": rows,
    }

    print("=" * 76)
    print("INDEPENDENT TORCH VERSUS HSOLVER SPECTRAL SUMMARY")
    print("=" * 76)
    print("This is a descriptive comparison: each backend used its own trajectory and EVD.")
    for row in rows:
        if row["trace_energy_threshold"] != 0.99:
            continue
        print("-" * 76)
        print(row["target_id"])
        print(
            "aggregate residual median Torch/Hsolver: "
            f"{row['torch_aggregate_residual_median']}/"
            f"{row['hsolver_aggregate_residual_median']}"
        )
        print(
            "99% leading subspace overlap median Torch/Hsolver: "
            f"{row['torch_leading_subspace_overlap_median']}/"
            f"{row['hsolver_leading_subspace_overlap_median']}"
        )
        print(
            "99% tail residual share median Torch/Hsolver: "
            f"{row['torch_tail_residual_share_median']}/"
            f"{row['hsolver_tail_residual_share_median']}"
        )

    if args.csv_output is not None:
        write_csv(args.csv_output, rows)
        print(f"Comparison CSV: {args.csv_output}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Comparison JSON: {args.json_output}")


if __name__ == "__main__":
    main()
