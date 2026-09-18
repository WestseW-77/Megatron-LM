#!/usr/bin/env python3
"""Compare matched Torch and Hsolver SOAP QR timing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_directory", type=Path)
    return parser.parse_args()


def read_summary(path: Path, expected_backend: str) -> dict:
    summary_path = path / "qr_timing_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_backend = summary["requested_qr_backend"]
    if actual_backend != expected_backend:
        raise RuntimeError(
            f"{summary_path} has requested backend {actual_backend!r}; expected {expected_backend!r}"
        )
    return summary


def mean(summary: dict, key: str) -> float:
    return float(summary[key]["mean"])


def main() -> None:
    args = parse_args()
    torch_summary = read_summary(args.pair_directory / "torch", "torch")
    hsolver_summary = read_summary(args.pair_directory / "hsolver", "hsolver")

    comparable_keys = (
        "global_batch_size",
        "gradient_accumulation_steps",
        "tokens_per_optimizer_step",
        "measurement_start_iteration",
        "measurement_end_iteration",
        "measurement_count",
    )
    mismatches = {
        key: (torch_summary[key], hsolver_summary[key])
        for key in comparable_keys
        if torch_summary[key] != hsolver_summary[key]
    }
    if mismatches:
        raise RuntimeError(f"Torch and Hsolver summaries are not comparable: {mismatches}")

    torch_step = mean(torch_summary, "step_ms")
    hsolver_step = mean(hsolver_summary, "step_ms")
    torch_optimizer = mean(torch_summary, "optimizer_ms")
    hsolver_optimizer = mean(hsolver_summary, "optimizer_ms")
    torch_qr = mean(torch_summary, "qr_ms_on_optimizer_critical_rank")
    hsolver_qr = mean(hsolver_summary, "qr_ms_on_optimizer_critical_rank")
    torch_non_qr = mean(torch_summary, "non_qr_optimizer_ms_on_optimizer_critical_rank")
    hsolver_non_qr = mean(hsolver_summary, "non_qr_optimizer_ms_on_optimizer_critical_rank")

    step_savings = torch_step - hsolver_step
    optimizer_savings = torch_optimizer - hsolver_optimizer
    qr_savings = torch_qr - hsolver_qr
    comparison = {
        "global_batch_size": torch_summary["global_batch_size"],
        "measurement_start_iteration": torch_summary["measurement_start_iteration"],
        "measurement_end_iteration": torch_summary["measurement_end_iteration"],
        "torch": {
            "step_mean_ms": torch_step,
            "optimizer_mean_ms": torch_optimizer,
            "qr_mean_ms_on_optimizer_critical_rank": torch_qr,
            "non_qr_optimizer_mean_ms_on_optimizer_critical_rank": torch_non_qr,
        },
        "hsolver": {
            "step_mean_ms": hsolver_step,
            "optimizer_mean_ms": hsolver_optimizer,
            "qr_mean_ms_on_optimizer_critical_rank": hsolver_qr,
            "non_qr_optimizer_mean_ms_on_optimizer_critical_rank": hsolver_non_qr,
        },
        "torch_over_hsolver_speedup": {
            "step": torch_step / hsolver_step,
            "optimizer": torch_optimizer / hsolver_optimizer,
            "qr": torch_qr / hsolver_qr,
        },
        "torch_minus_hsolver_ms": {
            "step": step_savings,
            "optimizer": optimizer_savings,
            "qr": qr_savings,
            "non_qr_optimizer": torch_non_qr - hsolver_non_qr,
        },
        "qr_savings_over_step_savings": qr_savings / step_savings,
        "qr_savings_over_optimizer_savings": qr_savings / optimizer_savings,
    }

    output_path = args.pair_directory / "qr_pair_comparison.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)

    print(f"GBS={comparison['global_batch_size']}")
    print(f"TORCH_STEP_MEAN_MS={torch_step:.3f}")
    print(f"HSOLVER_STEP_MEAN_MS={hsolver_step:.3f}")
    print(f"STEP_SPEEDUP={torch_step / hsolver_step:.6f}")
    print(f"TORCH_OPTIMIZER_MEAN_MS={torch_optimizer:.3f}")
    print(f"HSOLVER_OPTIMIZER_MEAN_MS={hsolver_optimizer:.3f}")
    print(f"OPTIMIZER_SPEEDUP={torch_optimizer / hsolver_optimizer:.6f}")
    print(f"TORCH_QR_MEAN_MS={torch_qr:.3f}")
    print(f"HSOLVER_QR_MEAN_MS={hsolver_qr:.3f}")
    print(f"QR_SPEEDUP={torch_qr / hsolver_qr:.6f}")
    print(f"QR_SAVINGS_OVER_STEP_SAVINGS={qr_savings / step_savings:.6f}")
    print(f"QR_SAVINGS_OVER_OPTIMIZER_SAVINGS={qr_savings / optimizer_savings:.6f}")


if __name__ == "__main__":
    main()
