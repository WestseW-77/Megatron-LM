#!/usr/bin/env python3
"""Build and validate the pinned Nemotron sample used by the SOAP 8B experiment."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


DATASET_REPO = "nvidia/Nemotron-Pretraining-Dataset-sample"
DATASET_REVISION = "3ad096e6394e487bb4f778733300da85275bb449"
DATASET_FILE = "Nemotron-CC-Translated-Diverse-QA/part_000000.parquet"
DATASET_SHA256 = "835851141069310879d95d49363d1c96e280a552ac4d4b53a6dc83c30e9e34a2"

TOKENIZER_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
TOKENIZER_REVISION = "97ab8012882a655dc38df4fee47422aca9caca07"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

EXPECTED = {
    "jsonl_sha256": "3eccdead5ea3b707be399b6ac3d693b9f1d70d22205e180aa8d163f687a550a1",
    "manifest_sha256": "c67262255c5e042648a268d6c6320331ceac553d49a3b4981ecc3f9bb5bf9d2c",
    "tokenizer_sha256": {
        "config.json": "c78db134b3aecd82042b9a573bd0d71acabfee3f1b4d082fe78d1c1d317cebfb",
        "derivation.json": "56473d1c61883e35d4ccaa318851a66214db110ce3b55e2fbea3805d5f50bbf4",
        "special_tokens_map.json": (
            "afe7d139eb6dc4aff93f89749945dce6fc059c22cbaa394724ae01b8dd40339d"
        ),
        "tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
        "tokenizer_config.json": "9a4c2659ce205101e66178b891f29d450c394edf3fa3c0944503379f3d0a0416",
    },
    "bin_sha256": "2ac059d4295f3c1e5600f4ab47d44bb22bac9717e655519166686de3f8e0fa3e",
    "idx_sha256": "4df050a7b382da46c20ebd80eb736c6cbad3ab510a5c3d0bc5c6d0fa7dd99e5d",
    "documents": 15441,
    "text_tokens": 14943607,
    "total_tokens": 14959048,
    "minimum_length": 11,
    "maximum_length": 4130,
    "minimum_token": 2,
    "maximum_token": 131070,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing required file: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def output_state(paths: list[Path], hashes: list[str]) -> str:
    existing = [path.exists() for path in paths]
    if not any(existing):
        return "missing"
    if not all(existing):
        raise RuntimeError(f"partial output exists; refusing to overwrite: {paths}")
    for path, expected in zip(paths, hashes, strict=True):
        require_hash(path, expected)
    return "valid"


def download_inputs(source_dir: Path, base_tokenizer_dir: Path, offline: bool) -> None:
    if not offline:
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=source_dir,
            allow_patterns=[DATASET_FILE, "README.md", "LICENSE.md"],
        )
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            repo_type="model",
            revision=TOKENIZER_REVISION,
            local_dir=base_tokenizer_dir,
            allow_patterns=list(TOKENIZER_FILES),
        )

    require_hash(source_dir / DATASET_FILE, DATASET_SHA256)
    for name in TOKENIZER_FILES:
        if not (base_tokenizer_dir / name).is_file():
            raise RuntimeError(f"missing tokenizer input: {base_tokenizer_dir / name}")

    weights = list(base_tokenizer_dir.glob("*.safetensors")) + list(
        base_tokenizer_dir.glob("*.bin")
    )
    if weights:
        raise RuntimeError(f"unexpected model weights in tokenizer directory: {weights}")


def derive_tokenizer(base_dir: Path, derived_dir: Path) -> None:
    names = list(EXPECTED["tokenizer_sha256"])
    paths = [derived_dir / name for name in names]
    hashes = [EXPECTED["tokenizer_sha256"][name] for name in names]
    if output_state(paths, hashes) == "valid":
        print("Derived tokenizer already exists and matches reference hashes.")
        return

    if derived_dir.exists() and any(derived_dir.iterdir()):
        raise RuntimeError(
            f"nonempty tokenizer output exists; refusing to overwrite: {derived_dir}"
        )
    derived_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, use_fast=True)
    original_vocab = tokenizer.get_vocab()
    if tokenizer.convert_tokens_to_ids("</s>") != 2:
        raise RuntimeError("base tokenizer does not map </s> to ID 2")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != 11:
        raise RuntimeError("base tokenizer does not map <|im_end|> to ID 11")

    tokenizer.eos_token = "</s>"
    tokenizer.save_pretrained(derived_dir)
    shutil.copy2(base_dir / "config.json", derived_dir / "config.json")

    manifest = {
        "operation": {
            "chat_end_token": "<|im_end|>",
            "chat_end_token_id": 11,
            "eos_token": "</s>",
            "eos_token_id": 2,
        },
        "purpose": (
            "Use the public Nemotron-3 Base vocabulary for plain next-token "
            "pretraining with document EOD ID 2."
        ),
        "source_repo": TOKENIZER_REPO,
        "source_revision": TOKENIZER_REVISION,
        "vocabulary_changed": False,
    }
    with (derived_dir / "derivation.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    reloaded = AutoTokenizer.from_pretrained(derived_dir, local_files_only=True, use_fast=True)
    if reloaded.eos_token_id != 2 or reloaded.get_vocab() != original_vocab:
        raise RuntimeError("derived tokenizer changed token IDs or has the wrong EOS")
    for path, expected in zip(paths, hashes, strict=True):
        require_hash(path, expected)


def convert_jsonl(source_path: Path, output_dir: Path) -> None:
    jsonl_path = output_dir / "nemotron_cc_translated_diverse_qa.jsonl"
    manifest_path = output_dir / "conversion_manifest.json"
    if (
        output_state(
            [jsonl_path, manifest_path],
            [EXPECTED["jsonl_sha256"], EXPECTED["manifest_sha256"]],
        )
        == "valid"
    ):
        print("Derived JSONL already exists and matches reference hashes.")
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"nonempty JSONL output exists; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(source_path, columns=["id", "language", "text"])
    rows = table.to_pylist()
    identifiers: set[str] = set()
    texts: set[str] = set()
    duplicate_text_rows = 0
    nonempty_rows = 0
    language_counts: collections.Counter[str] = collections.Counter()

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            record = {"id": row["id"], "language": row["language"], "text": row["text"]}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            identifiers.add(row["id"])
            language_counts[row["language"]] += 1
            nonempty_rows += bool(row["text"])
            if row["text"] in texts:
                duplicate_text_rows += 1
            else:
                texts.add(row["text"])

    manifest = {
        "conversion": {
            "deduplicate": False,
            "included_fields": ["id", "language", "text"],
            "json_ensure_ascii": False,
            "json_separators": [",", ":"],
            "line_ending": "LF",
            "preserve_source_row_order": True,
            "shuffle": False,
        },
        "format_version": 1,
        "output": {
            "bytes": jsonl_path.stat().st_size,
            "duplicate_text_rows": duplicate_text_rows,
            "file": jsonl_path.name,
            "first_id": rows[0]["id"],
            "language_counts": dict(sorted(language_counts.items())),
            "last_id": rows[-1]["id"],
            "nonempty_rows": nonempty_rows,
            "rows": len(rows),
            "sha256": sha256(jsonl_path),
            "unique_ids": len(identifiers),
        },
        "source": {
            "file": DATASET_FILE,
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "sha256": DATASET_SHA256,
            "subset": "Nemotron-CC-Translated-Diverse-QA",
        },
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    require_hash(jsonl_path, EXPECTED["jsonl_sha256"])
    require_hash(manifest_path, EXPECTED["manifest_sha256"])


def preprocess_megatron(
    megatron_root: Path, jsonl_path: Path, tokenizer_dir: Path, output_prefix: Path, workers: int
) -> None:
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    if output_state(
        [bin_path, idx_path], [EXPECTED["bin_sha256"], EXPECTED["idx_sha256"]]
    ) == "valid":
        print("Megatron indexed dataset already exists and matches reference hashes.")
        return

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        sys.executable,
        str(megatron_root / "tools/preprocess_data.py"),
        "--input",
        str(jsonl_path),
        "--json-keys",
        "text",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_dir),
        "--append-eod",
        "--output-prefix",
        str(output_prefix).removesuffix("_text_document"),
        "--workers",
        str(workers),
        "--partitions",
        "1",
        "--keep-sequential-samples",
        "--log-interval",
        "500",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=megatron_root, env=env, check=True)
    require_hash(bin_path, EXPECTED["bin_sha256"])
    require_hash(idx_path, EXPECTED["idx_sha256"])


def validate_indexed_dataset(megatron_root: Path, output_prefix: Path) -> None:
    sys.path.insert(0, str(megatron_root))
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    dataset = IndexedDataset(str(output_prefix))
    lengths = dataset.sequence_lengths
    documents = len(dataset.document_indices) - 1
    total_tokens = int(lengths.sum())
    text_tokens = total_tokens - documents
    minimum_token = None
    maximum_token = None
    bad_eod = 0

    for index in range(len(dataset)):
        tokens = np.asarray(dataset[index])
        current_min = int(tokens.min())
        current_max = int(tokens.max())
        minimum_token = current_min if minimum_token is None else min(minimum_token, current_min)
        maximum_token = current_max if maximum_token is None else max(maximum_token, current_max)
        bad_eod += int(tokens[-1]) != 2

    actual = {
        "documents": documents,
        "sequences": len(dataset),
        "text_tokens": text_tokens,
        "total_tokens": total_tokens,
        "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()),
        "minimum_token": minimum_token,
        "maximum_token": maximum_token,
        "bad_eod_documents": bad_eod,
    }
    print(json.dumps(actual, indent=2, sort_keys=True))

    for key in (
        "documents",
        "text_tokens",
        "total_tokens",
        "minimum_length",
        "maximum_length",
        "minimum_token",
        "maximum_token",
    ):
        if actual[key] != EXPECTED[key]:
            raise RuntimeError(
                f"indexed dataset {key}: expected {EXPECTED[key]}, got {actual[key]}"
            )
    if actual["sequences"] != EXPECTED["documents"] or bad_eod:
        raise RuntimeError("indexed dataset has invalid sequence count or EOD placement")
    expected_boundaries = np.arange(EXPECTED["documents"] + 1, dtype=dataset.document_indices.dtype)
    if not np.array_equal(dataset.document_indices, expected_boundaries):
        raise RuntimeError("indexed dataset document boundaries do not match one sequence per row")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    megatron_root = Path(__file__).resolve().parents[3]

    source_dir = data_root / "sources/nemotron-pretraining-dataset-sample"
    base_tokenizer_dir = data_root / "tokenizers/nemotron-3-nano-30b-a3b-base"
    derived_tokenizer_dir = (
        data_root / "tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
    )
    jsonl_dir = data_root / "derived/nemotron-cc-translated-diverse-qa"
    output_prefix = (
        data_root
        / "processed/nemotron-cc-translated-diverse-qa-eod2"
        / "nemotron_cc_translated_diverse_qa_eod2_text_document"
    )

    download_inputs(source_dir, base_tokenizer_dir, args.offline)
    derive_tokenizer(base_tokenizer_dir, derived_tokenizer_dir)
    convert_jsonl(source_dir / DATASET_FILE, jsonl_dir)
    preprocess_megatron(
        megatron_root,
        jsonl_dir / "nemotron_cc_translated_diverse_qa.jsonl",
        derived_tokenizer_dir,
        output_prefix,
        args.workers,
    )
    validate_indexed_dataset(megatron_root, output_prefix)
    print("NEMOTRON SAMPLE DATA PREPARATION: PASS")


if __name__ == "__main__":
    main()
