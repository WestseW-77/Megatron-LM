#!/usr/bin/env python3
"""Flatten SOAP parameter inventory records into an auditable candidate table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    return parser.parse_args()


def load(input_directory: Path) -> list[dict[str, Any]]:
    files = sorted(input_directory.glob("rank_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No rank_*.jsonl files found under {input_directory}")
    result = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
                if item.get("event") == "parameter_inventory":
                    result.append(item)
    if not result:
        raise ValueError(f"No parameter_inventory records found under {input_directory}")
    return result


def flatten(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for axis in record["axes"]:
            rows.append(
                {
                    "rank": record["rank"],
                    "pipeline_parallel_rank": record["pipeline_parallel_rank"],
                    "tensor_parallel_rank": record["tensor_parallel_rank"],
                    "group_index": record["group_index"],
                    "param_index": record["param_index"],
                    "parameter_name": record["parameter_name"],
                    "parameter_shape": "x".join(
                        str(value) for value in record["parameter_shape"]
                    ),
                    "axis_index": axis["axis_index"],
                    "axis_name": axis["axis_name"],
                    "dimension": axis["dimension"],
                    "factor_shape": "x".join(str(value) for value in axis["shape"]),
                    "dtype": axis["dtype"],
                    "device": axis["device"],
                    "hsolver_supported": axis["hsolver_supported"],
                }
            )
    rows.sort(
        key=lambda row: (
            row["pipeline_parallel_rank"],
            row["tensor_parallel_rank"],
            row["parameter_name"],
            row["axis_index"],
        )
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
    rows = flatten(load(args.input))
    candidates = [row for row in rows if row["hsolver_supported"]]
    if not candidates:
        raise ValueError("Inventory contains no Hsolver-supported factor axes")

    write_csv(args.csv_output, rows)
    write_csv(args.candidates_output, candidates)

    print("=" * 72)
    print("SOAP EIGENBASIS TARGET INVENTORY")
    print("=" * 72)
    print(f"all factor axes: {len(rows)}")
    print(f"Hsolver-supported axes: {len(candidates)}")
    print("supported axes by (pipeline rank, tensor rank, dimension):")
    counts = Counter(
        (
            row["pipeline_parallel_rank"],
            row["tensor_parallel_rank"],
            row["dimension"],
        )
        for row in candidates
    )
    for key, count in sorted(counts.items()):
        print(f"  pp={key[0]} tp={key[1]} dim={key[2]} count={count}")
    print(f"Full inventory CSV: {args.csv_output}")
    print(f"Hsolver candidate CSV: {args.candidates_output}")


if __name__ == "__main__":
    main()
