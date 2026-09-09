#!/usr/bin/env python3
"""Build a deterministic 100B-token sample of Nemotron-CC high/actual data."""

from __future__ import annotations

import argparse
import collections
import fractions
import gzip
import hashlib
import http.client
import importlib.metadata
import io
import json
import multiprocessing
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


DATASET_INDEX_URL = (
    "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
)
DATASET_BASE_URL = "https://data.commoncrawl.org/"
DATASET_INDEX_SHA256 = "d8201c1e05b5e9fecef7678457c172f98a19a25f1dd656cdba1f937e7a29e68e"
PARTITION_PREFIX = (
    "contrib/Nemotron/Nemotron-CC/data-jsonl/"
    "quality=high/kind=actual/kind2=actual/"
)
EXPECTED_PARTITION_FILES = 2755
EXPECTED_SNAPSHOTS = 99
PUBLISHED_PARTITION_TOKENS = 553_000_000_000
DEFAULT_TARGET_TEXT_TOKENS = 100_000_000_000
SAMPLING_NAMESPACE = "hsolver-nemotron-100b-v1"
SAMPLING_ORDER_SHA256 = "e693749197c2899ebabb27c4f44024d823425c69e372031d409c986ac64d94d4"
SNAPSHOT_PATTERN = re.compile(r"/(CC-MAIN-\d{4}-\d{2})-part-\d+\.jsonl\.zstd$")

TOKENIZER_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
TOKENIZER_REVISION = "97ab8012882a655dc38df4fee47422aca9caca07"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
EXPECTED_DERIVED_TOKENIZER_SHA256 = {
    "config.json": "c78db134b3aecd82042b9a573bd0d71acabfee3f1b4d082fe78d1c1d317cebfb",
    "derivation.json": "56473d1c61883e35d4ccaa318851a66214db110ce3b55e2fbea3805d5f50bbf4",
    "special_tokens_map.json": "afe7d139eb6dc4aff93f89749945dce6fc059c22cbaa394724ae01b8dd40339d",
    "tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    "tokenizer_config.json": "9a4c2659ce205101e66178b891f29d450c394edf3fa3c0944503379f3d0a0416",
}

_WORKER_TOKENIZER: Any = None
_WORKER_EOD_ID: int | None = None


@dataclass(frozen=True)
class Candidate:
    """One source shard in the deterministic stratified sampling order."""

    path: str
    selection_hash: str
    snapshot: str
    snapshot_count: int
    snapshot_rank: int


@dataclass(frozen=True)
class DownloadInfo:
    """Metadata observed while downloading a source object."""

    bytes: int
    etag: str | None
    last_modified: str | None


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    """Require a file to exist and have the expected SHA256 digest."""

    if not path.is_file():
        raise RuntimeError(f"missing required file: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def write_json(path: Path, value: Any) -> None:
    """Atomically write a stable JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_lines(path: Path, lines: list[str]) -> None:
    """Atomically write newline-terminated text lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    temporary.replace(path)


def build_url_opener(ignore_proxy: bool) -> urllib.request.OpenerDirector:
    """Build an opener that either honors or explicitly ignores proxy variables."""

    if ignore_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def download_file(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
    retries: int,
    timeout: int,
) -> DownloadInfo:
    """Download one object atomically, resuming an existing partial download."""

    if destination.is_file():
        return DownloadInfo(destination.stat().st_size, None, None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "hsolver-nemotron-preparation/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with opener.open(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"

                content_range = response.headers.get("Content-Range")
                if content_range and "/" in content_range:
                    expected_size = int(content_range.rsplit("/", 1)[1])
                else:
                    content_length = response.headers.get("Content-Length")
                    expected_size = offset + int(content_length) if content_length else None

                downloaded = offset
                previous_report = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - previous_report >= 30:
                            if expected_size:
                                percent = 100.0 * downloaded / expected_size
                                print(
                                    f"  download: {downloaded / 1e9:.2f}/{expected_size / 1e9:.2f} GB "
                                    f"({percent:.1f}%)",
                                    flush=True,
                                )
                            else:
                                print(f"  download: {downloaded / 1e9:.2f} GB", flush=True)
                            previous_report = now

                actual_size = partial.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    raise OSError(
                        f"incomplete download for {url}: expected {expected_size}, got {actual_size}"
                    )
                partial.replace(destination)
                return DownloadInfo(
                    bytes=actual_size,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
        except (OSError, TimeoutError, http.client.HTTPException, urllib.error.URLError) as error:
            last_error = error
            if attempt == retries:
                break
            delay = min(60, 2**attempt)
            print(
                f"  download attempt {attempt}/{retries} failed: {error}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_error


def prepare_source_index(
    data_root: Path,
    opener: urllib.request.OpenerDirector,
    offline: bool,
    retries: int,
    timeout: int,
) -> Path:
    """Download and validate the pinned Common Crawl object-path index."""

    index_path = data_root / "sources/nemotron-cc/data-jsonl.paths.gz"
    if not index_path.exists():
        if offline:
            raise RuntimeError(f"offline mode requires the source index: {index_path}")
        print(f"Downloading source index: {DATASET_INDEX_URL}", flush=True)
        download_file(opener, DATASET_INDEX_URL, index_path, retries, timeout)
    require_sha256(index_path, DATASET_INDEX_SHA256)
    return index_path


def load_partition_paths(index_path: Path) -> list[str]:
    """Read and validate all paths in the pinned high/actual partition."""

    with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
        paths = [line.rstrip("\n") for line in handle if line.startswith(PARTITION_PREFIX)]

    if len(paths) != EXPECTED_PARTITION_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_PARTITION_FILES} partition files, found {len(paths)}"
        )
    if len(set(paths)) != len(paths):
        raise RuntimeError("source index contains duplicate paths in the selected partition")
    return paths


def build_sampling_order(paths: list[str]) -> list[Candidate]:
    """Build a deterministic order whose prefixes preserve snapshot proportions."""

    grouped: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for path in paths:
        match = SNAPSHOT_PATTERN.search(path)
        if match is None:
            raise RuntimeError(f"cannot extract Common Crawl snapshot from path: {path}")
        selection_hash = hashlib.sha256(
            f"{SAMPLING_NAMESPACE}\0{path}".encode("utf-8")
        ).hexdigest()
        grouped[match.group(1)].append((selection_hash, path))

    if len(grouped) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(f"expected {EXPECTED_SNAPSHOTS} snapshots, found {len(grouped)}")

    sortable: list[tuple[fractions.Fraction, str, str, Candidate]] = []
    for snapshot, entries in sorted(grouped.items()):
        entries.sort()
        count = len(entries)
        for zero_based_rank, (selection_hash, path) in enumerate(entries):
            candidate = Candidate(
                path=path,
                selection_hash=selection_hash,
                snapshot=snapshot,
                snapshot_count=count,
                snapshot_rank=zero_based_rank + 1,
            )
            quantile = fractions.Fraction(2 * zero_based_rank + 1, 2 * count)
            sortable.append((quantile, selection_hash, path, candidate))

    sortable.sort(key=lambda item: item[:3])
    return [item[3] for item in sortable]


def write_sampling_plan(
    manifest_dir: Path,
    index_path: Path,
    candidates: list[Candidate],
    target_text_tokens: int,
) -> None:
    """Write deterministic source and sampling-order metadata."""

    order_path = manifest_dir / "sampling_order.jsonl"
    rows = []
    for order, candidate in enumerate(candidates, start=1):
        row = {"order": order, **asdict(candidate)}
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    write_lines(order_path, rows)
    require_sha256(order_path, SAMPLING_ORDER_SHA256)

    estimated_shards = (target_text_tokens * len(candidates) + PUBLISHED_PARTITION_TOKENS - 1) // (
        PUBLISHED_PARTITION_TOKENS
    )
    plan = {
        "format_version": 1,
        "partition": {
            "expected_files": EXPECTED_PARTITION_FILES,
            "expected_snapshots": EXPECTED_SNAPSHOTS,
            "prefix": PARTITION_PREFIX,
            "published_text_tokens": PUBLISHED_PARTITION_TOKENS,
        },
        "sampling": {
            "algorithm": (
                "Within each snapshot, sort paths by SHA256(namespace + NUL + path). "
                "Assign rank midpoint (2*rank-1)/(2*count), then globally sort by that "
                "fraction, selection hash, and path. Consume complete shards in this order."
            ),
            "estimated_shards": estimated_shards,
            "namespace": SAMPLING_NAMESPACE,
            "order_file": order_path.name,
            "order_file_sha256": SAMPLING_ORDER_SHA256,
            "stop_rule": (
                "Stop after the first complete shard that makes cumulative tokenizer text "
                "tokens greater than or equal to target_text_tokens. Never truncate documents."
            ),
            "target_text_tokens": target_text_tokens,
        },
        "source_index": {
            "file": index_path.name,
            "sha256": DATASET_INDEX_SHA256,
            "url": DATASET_INDEX_URL,
        },
    }
    write_json(manifest_dir / "sampling_plan.json", plan)


def prepare_tokenizer(data_root: Path, offline: bool) -> Path:
    """Download and derive the pinned plain-pretraining tokenizer."""

    base_dir = data_root / "tokenizers/nemotron-3-nano-30b-a3b-base"
    derived_dir = data_root / "tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
    derived_paths = [derived_dir / name for name in EXPECTED_DERIVED_TOKENIZER_SHA256]

    if any(path.exists() for path in derived_paths):
        if not all(path.is_file() for path in derived_paths):
            raise RuntimeError(f"partial derived tokenizer exists: {derived_dir}")
        for name, expected in EXPECTED_DERIVED_TOKENIZER_SHA256.items():
            require_sha256(derived_dir / name, expected)
        return derived_dir

    base_paths = [base_dir / name for name in TOKENIZER_FILES]
    if not all(path.is_file() for path in base_paths):
        if offline:
            raise RuntimeError(f"offline mode requires tokenizer inputs in: {base_dir}")
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            repo_type="model",
            revision=TOKENIZER_REVISION,
            local_dir=base_dir,
            allow_patterns=list(TOKENIZER_FILES),
        )
    for name in TOKENIZER_FILES:
        if not (base_dir / name).is_file():
            raise RuntimeError(f"missing tokenizer input: {base_dir / name}")

    from prepare_nemotron_sample import derive_tokenizer

    derive_tokenizer(base_dir, derived_dir)
    for name, expected in EXPECTED_DERIVED_TOKENIZER_SHA256.items():
        require_sha256(derived_dir / name, expected)
    return derived_dir


def initialize_tokenizer_worker(tokenizer_dir: str, eod_id: int) -> None:
    """Initialize one tokenizer instance per multiprocessing worker."""

    global _WORKER_EOD_ID, _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, use_fast=True
    )
    _WORKER_EOD_ID = eod_id


def encode_record(item: tuple[int, str]) -> tuple[np.ndarray, int, str]:
    """Parse and tokenize one JSONL record while preserving source order."""

    if _WORKER_TOKENIZER is None or _WORKER_EOD_ID is None:
        raise RuntimeError("tokenizer worker was not initialized")
    line_number, line = item
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON on source line {line_number}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"source line {line_number} is not a JSON object")
    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"source line {line_number} has no string 'text' field")

    token_ids = _WORKER_TOKENIZER(text).input_ids
    text_tokens = len(token_ids)
    if token_ids:
        token_ids.append(_WORKER_EOD_ID)
    language = record.get("language", "unknown")
    if not isinstance(language, str):
        language = "unknown"
    return np.asarray(token_ids, dtype=np.int32), text_tokens, language


def iter_zstd_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    """Yield numbered UTF-8 lines from a Zstandard-compressed JSONL file."""

    with pa.input_stream(str(path)) as source:
        with pa.CompressedInputStream(source, "zstd") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                for line_number, line in enumerate(text, start=1):
                    yield line_number, line


def shard_names(candidate: Candidate) -> tuple[str, str]:
    """Return a stable shard stem and source filename."""

    source_name = Path(candidate.path).name
    suffix = ".jsonl.zstd"
    if not source_name.endswith(suffix):
        raise RuntimeError(f"unexpected source suffix: {source_name}")
    return source_name.removesuffix(suffix), source_name


def validate_existing_shard(
    candidate: Candidate,
    metadata_path: Path,
    processed_dir: Path,
    tokenizer_sha256: str,
    verify_hashes: bool,
) -> dict[str, Any] | None:
    """Validate and return completed shard metadata, or return None if absent."""

    stem, _ = shard_names(candidate)
    prefix = processed_dir / "shards" / f"{stem}_text_document"
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")

    if not metadata_path.exists():
        if bin_path.exists() or idx_path.exists():
            raise RuntimeError(
                f"indexed output exists without completion metadata: {prefix}; "
                "move the orphaned files aside before retrying"
            )
        return None

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["source"]["path"] != candidate.path:
        raise RuntimeError(f"source path mismatch in {metadata_path}")
    if metadata["sampling"] != asdict(candidate):
        raise RuntimeError(f"sampling metadata mismatch in {metadata_path}")
    if metadata["tokenizer"]["tokenizer_json_sha256"] != tokenizer_sha256:
        raise RuntimeError(f"tokenizer mismatch in {metadata_path}")

    for output_path, key in ((bin_path, "bin"), (idx_path, "idx")):
        expected = metadata["indexed"][key]
        if not output_path.is_file() or output_path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"missing or wrong-sized indexed output: {output_path}")
        if verify_hashes:
            require_sha256(output_path, expected["sha256"])
    return metadata


def process_shard(
    megatron_root: Path,
    candidate: Candidate,
    compressed_path: Path,
    download_info: DownloadInfo,
    source_sha256: str,
    processed_dir: Path,
    metadata_path: Path,
    tokenizer_sha256: str,
    tokenizer_vocabulary_size: int,
    eod_id: int,
    pool: Pool,
    chunksize: int,
    log_interval: int,
) -> dict[str, Any]:
    """Tokenize one complete source shard into an atomic Megatron indexed dataset."""

    sys.path.insert(0, str(megatron_root))
    from megatron.core.datasets.indexed_dataset import DType, IndexedDataset, IndexedDatasetBuilder

    dtype = DType.optimal_dtype(tokenizer_vocabulary_size)
    stem, _ = shard_names(candidate)
    output_prefix = processed_dir / "shards" / f"{stem}_text_document"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary_prefix = output_prefix.with_name(f".{output_prefix.name}.building")
    temporary_bin = Path(f"{temporary_prefix}.bin")
    temporary_idx = Path(f"{temporary_prefix}.idx")
    final_bin = Path(f"{output_prefix}.bin")
    final_idx = Path(f"{output_prefix}.idx")

    for temporary in (temporary_bin, temporary_idx):
        if temporary.exists():
            temporary.unlink()

    builder = IndexedDatasetBuilder(str(temporary_bin), dtype=dtype)
    documents = 0
    empty_documents = 0
    text_tokens = 0
    total_tokens = 0
    minimum_text_tokens: int | None = None
    maximum_text_tokens = 0
    minimum_token_id: int | None = None
    maximum_token_id: int | None = None
    languages: collections.Counter[str] = collections.Counter()
    start = time.monotonic()

    try:
        encoded = pool.imap(encode_record, iter_zstd_jsonl(compressed_path), chunksize)
        for token_ids, document_text_tokens, language in encoded:
            documents += 1
            languages[language] += 1
            text_tokens += document_text_tokens
            total_tokens += int(token_ids.size)
            minimum_text_tokens = (
                document_text_tokens
                if minimum_text_tokens is None
                else min(minimum_text_tokens, document_text_tokens)
            )
            maximum_text_tokens = max(maximum_text_tokens, document_text_tokens)

            if token_ids.size:
                current_min = int(token_ids.min())
                current_max = int(token_ids.max())
                minimum_token_id = (
                    current_min if minimum_token_id is None else min(minimum_token_id, current_min)
                )
                maximum_token_id = (
                    current_max if maximum_token_id is None else max(maximum_token_id, current_max)
                )
                builder.add_document(token_ids, [int(token_ids.size)])
            else:
                empty_documents += 1
                builder.add_document(token_ids, [])

            if documents % log_interval == 0:
                elapsed = time.monotonic() - start
                print(
                    f"  tokenize: documents={documents:,} text_tokens={text_tokens:,} "
                    f"rate={text_tokens / max(elapsed, 1e-9) / 1e6:.2f} Mtok/s",
                    flush=True,
                )

        builder.finalize(str(temporary_idx))
    except BaseException:
        if not builder.data_file.closed:
            builder.data_file.close()
        for temporary in (temporary_bin, temporary_idx):
            temporary.unlink(missing_ok=True)
        raise

    indexed = IndexedDataset(str(temporary_prefix))
    indexed_documents = len(indexed.document_indices) - 1
    indexed_total_tokens = int(indexed.sequence_lengths.sum())
    indexed_sequences = len(indexed)
    del indexed
    if indexed_documents != documents or indexed_total_tokens != total_tokens:
        raise RuntimeError(
            f"indexed validation failed for {candidate.path}: "
            f"documents={indexed_documents}/{documents}, tokens={indexed_total_tokens}/{total_tokens}"
        )
    if indexed_sequences != documents - empty_documents:
        raise RuntimeError(f"indexed sequence count mismatch for {candidate.path}")

    bin_sha256 = file_sha256(temporary_bin)
    idx_sha256 = file_sha256(temporary_idx)
    temporary_bin.replace(final_bin)
    temporary_idx.replace(final_idx)

    metadata = {
        "format_version": 1,
        "indexed": {
            "bin": {
                "bytes": final_bin.stat().st_size,
                "path": str(final_bin.relative_to(processed_dir)),
                "sha256": bin_sha256,
            },
            "idx": {
                "bytes": final_idx.stat().st_size,
                "path": str(final_idx.relative_to(processed_dir)),
                "sha256": idx_sha256,
            },
            "prefix": str(output_prefix.relative_to(processed_dir)),
            "token_dtype": np.dtype(dtype).name,
        },
        "sampling": asdict(candidate),
        "source": {
            "bytes": compressed_path.stat().st_size,
            "etag": download_info.etag,
            "last_modified": download_info.last_modified,
            "path": candidate.path,
            "sha256": source_sha256,
            "url": f"{DATASET_BASE_URL}{candidate.path}",
        },
        "statistics": {
            "documents": documents,
            "empty_documents": empty_documents,
            "eod_tokens": documents - empty_documents,
            "languages": dict(sorted(languages.items())),
            "maximum_text_tokens_per_document": maximum_text_tokens,
            "maximum_token_id": maximum_token_id,
            "minimum_text_tokens_per_document": minimum_text_tokens,
            "minimum_token_id": minimum_token_id,
            "sequences": indexed_sequences,
            "text_tokens": text_tokens,
            "total_tokens": total_tokens,
        },
        "tokenizer": {
            "append_eod": True,
            "eod_id": eod_id,
            "include_special_tokens": True,
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": tokenizer_sha256,
            "vocabulary_size": tokenizer_vocabulary_size,
        },
    }
    write_json(metadata_path, metadata)
    print(
        f"  complete: documents={documents:,} text_tokens={text_tokens:,} "
        f"indexed={final_bin.stat().st_size / 1e9:.2f} GB",
        flush=True,
    )
    return metadata


def write_final_manifest(
    megatron_root: Path,
    data_root: Path,
    manifest_dir: Path,
    processed_dir: Path,
    target_text_tokens: int,
    selected: list[tuple[Candidate, dict[str, Any]]],
    tokenizer_dir: Path,
) -> None:
    """Write selected paths, Megatron data arguments, and aggregate statistics."""

    selected_paths = [candidate.path for candidate, _ in selected]
    relative_prefixes = [metadata["indexed"]["prefix"] for _, metadata in selected]
    absolute_prefixes = [str((processed_dir / prefix).resolve()) for prefix in relative_prefixes]
    if any(any(character.isspace() for character in prefix) for prefix in absolute_prefixes):
        raise RuntimeError("Megatron --data-args-path does not support whitespace in data paths")
    write_lines(manifest_dir / "selected_paths.txt", selected_paths)
    write_lines(manifest_dir / "data_prefixes.relative.txt", relative_prefixes)
    write_lines(processed_dir / "data_args.txt", absolute_prefixes)

    selected_rows = []
    for candidate, metadata in selected:
        selected_rows.append(
            json.dumps(
                {
                    "indexed": metadata["indexed"],
                    "sampling": asdict(candidate),
                    "source": metadata["source"],
                    "statistics": metadata["statistics"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    selected_shards_path = manifest_dir / "selected_shards.jsonl"
    write_lines(selected_shards_path, selected_rows)

    aggregate = collections.Counter()
    language_counts: collections.Counter[str] = collections.Counter()
    snapshot_counts: collections.Counter[str] = collections.Counter()
    for candidate, metadata in selected:
        statistics = metadata["statistics"]
        aggregate.update(
            {
                "compressed_bytes": metadata["source"]["bytes"],
                "documents": statistics["documents"],
                "empty_documents": statistics["empty_documents"],
                "eod_tokens": statistics["eod_tokens"],
                "indexed_bin_bytes": metadata["indexed"]["bin"]["bytes"],
                "indexed_idx_bytes": metadata["indexed"]["idx"]["bytes"],
                "sequences": statistics["sequences"],
                "text_tokens": statistics["text_tokens"],
                "total_tokens": statistics["total_tokens"],
            }
        )
        language_counts.update(statistics["languages"])
        snapshot_counts[candidate.snapshot] += 1

    manifest = {
        "dataset": {
            "description": (
                "Deterministic stratified sample of Nemotron-CC high-quality real text"
            ),
            "partition_prefix": PARTITION_PREFIX,
            "published_partition_text_tokens": PUBLISHED_PARTITION_TOKENS,
        },
        "format_version": 1,
        "megatron": {
            "commit": get_git_commit(megatron_root),
            "data_args_file": str((processed_dir / "data_args.txt").relative_to(data_root)),
            "indexed_root": str(processed_dir.relative_to(data_root)),
        },
        "sampling": {
            "namespace": SAMPLING_NAMESPACE,
            "overshoot_text_tokens": aggregate["text_tokens"] - target_text_tokens,
            "sampling_order_sha256": SAMPLING_ORDER_SHA256,
            "selected_paths_file": "selected_paths.txt",
            "selected_paths_sha256": file_sha256(manifest_dir / "selected_paths.txt"),
            "selected_shards": len(selected),
            "selected_shards_file": selected_shards_path.name,
            "selected_shards_sha256": file_sha256(selected_shards_path),
            "snapshot_shard_counts": dict(sorted(snapshot_counts.items())),
            "target_text_tokens": target_text_tokens,
        },
        "software": {
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "python": sys.version.split()[0],
            "tokenizers": importlib.metadata.version("tokenizers"),
            "transformers": importlib.metadata.version("transformers"),
            "preparation_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "statistics": {
            **dict(aggregate),
            "languages": dict(sorted(language_counts.items())),
        },
        "tokenizer": {
            "directory": str(tokenizer_dir.relative_to(data_root)),
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": file_sha256(tokenizer_dir / "tokenizer.json"),
        },
    }
    write_json(manifest_dir / "dataset_manifest.json", manifest)


def get_git_commit(repository: Path) -> str | None:
    """Read a repository HEAD without making Git a runtime requirement."""

    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def target_label(target_text_tokens: int) -> str:
    """Return a stable filesystem label for a text-token target."""

    if target_text_tokens % 1_000_000_000 == 0:
        return f"{target_text_tokens // 1_000_000_000}b"
    return f"{target_text_tokens}-tokens"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="dataset workspace root")
    parser.add_argument(
        "--target-text-tokens",
        type=int,
        default=DEFAULT_TARGET_TEXT_TOKENS,
        help="minimum text tokens before appended EOD tokens (default: 100B)",
    )
    parser.add_argument("--workers", type=int, default=8, help="tokenizer worker processes")
    parser.add_argument("--chunksize", type=int, default=32, help="records per worker task")
    parser.add_argument(
        "--log-interval", type=int, default=10_000, help="documents between progress reports"
    )
    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--download-timeout", type=int, default=120, help="socket timeout in seconds")
    parser.add_argument("--offline", action="store_true", help="prohibit all downloads")
    parser.add_argument(
        "--ignore-proxy",
        action="store_true",
        help="bypass proxy variables for Common Crawl downloads only",
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="write the sampling order without dataset shards"
    )
    parser.add_argument(
        "--remove-compressed-after-indexing",
        action="store_true",
        help="delete a source shard only after its indexed output is complete",
    )
    parser.add_argument(
        "--verify-existing-hashes",
        action="store_true",
        help="rehash all reused indexed shard files instead of checking size only",
    )
    args = parser.parse_args()

    for name in (
        "target_text_tokens",
        "workers",
        "chunksize",
        "log_interval",
        "download_retries",
        "download_timeout",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    """Prepare the selected source shards and their Megatron indexes."""

    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    megatron_root = Path(__file__).resolve().parents[3]
    label = target_label(args.target_text_tokens)
    manifest_dir = data_root / f"manifests/nemotron-cc-high-actual-{label}"
    compressed_dir = data_root / "sources/nemotron-cc/high-actual"
    processed_dir = data_root / f"processed/nemotron-cc-high-actual-{label}-eod2"
    opener = build_url_opener(args.ignore_proxy)

    index_path = prepare_source_index(
        data_root, opener, args.offline, args.download_retries, args.download_timeout
    )
    candidates = build_sampling_order(load_partition_paths(index_path))
    write_sampling_plan(
        manifest_dir, index_path, candidates, args.target_text_tokens
    )
    print(
        f"Sampling plan: files={len(candidates):,} snapshots={EXPECTED_SNAPSHOTS} "
        f"target={args.target_text_tokens:,} text tokens",
        flush=True,
    )
    print(f"Plan directory: {manifest_dir}", flush=True)
    if args.plan_only:
        print("NEMOTRON-CC HIGH/ACTUAL SAMPLING PLAN: PASS")
        return

    tokenizer_dir = prepare_tokenizer(data_root, args.offline)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id != 2:
        raise RuntimeError(f"expected tokenizer EOD ID 2, got {tokenizer.eos_token_id}")
    tokenizer_sha256 = file_sha256(tokenizer_dir / "tokenizer.json")

    context = multiprocessing.get_context("spawn")
    selected: list[tuple[Candidate, dict[str, Any]]] = []
    cumulative_text_tokens = 0
    with context.Pool(
        args.workers,
        initializer=initialize_tokenizer_worker,
        initargs=(str(tokenizer_dir), tokenizer.eos_token_id),
    ) as pool:
        for order, candidate in enumerate(candidates, start=1):
            stem, source_name = shard_names(candidate)
            metadata_path = processed_dir / "metadata" / f"{stem}.json"
            metadata = validate_existing_shard(
                candidate,
                metadata_path,
                processed_dir,
                tokenizer_sha256,
                args.verify_existing_hashes,
            )

            compressed_path = compressed_dir / source_name
            if metadata is None:
                if not compressed_path.exists():
                    if args.offline:
                        raise RuntimeError(f"offline mode requires source shard: {compressed_path}")
                    print(
                        f"[{order}/{len(candidates)}] Downloading {candidate.path}", flush=True
                    )
                    download_info = download_file(
                        opener,
                        f"{DATASET_BASE_URL}{candidate.path}",
                        compressed_path,
                        args.download_retries,
                        args.download_timeout,
                    )
                else:
                    download_info = DownloadInfo(compressed_path.stat().st_size, None, None)

                print(f"[{order}/{len(candidates)}] Hashing {source_name}", flush=True)
                source_sha256 = file_sha256(compressed_path)
                print(f"[{order}/{len(candidates)}] Tokenizing {source_name}", flush=True)
                metadata = process_shard(
                    megatron_root=megatron_root,
                    candidate=candidate,
                    compressed_path=compressed_path,
                    download_info=download_info,
                    source_sha256=source_sha256,
                    processed_dir=processed_dir,
                    metadata_path=metadata_path,
                    tokenizer_sha256=tokenizer_sha256,
                    tokenizer_vocabulary_size=len(tokenizer),
                    eod_id=tokenizer.eos_token_id,
                    pool=pool,
                    chunksize=args.chunksize,
                    log_interval=args.log_interval,
                )
            else:
                print(f"[{order}/{len(candidates)}] Reusing indexed shard {source_name}", flush=True)

            if args.remove_compressed_after_indexing and compressed_path.exists():
                compressed_path.unlink()

            selected.append((candidate, metadata))
            cumulative_text_tokens += metadata["statistics"]["text_tokens"]
            print(
                f"  cumulative: shards={len(selected):,} "
                f"text_tokens={cumulative_text_tokens:,}/{args.target_text_tokens:,}",
                flush=True,
            )
            if cumulative_text_tokens >= args.target_text_tokens:
                break

    if cumulative_text_tokens < args.target_text_tokens:
        raise RuntimeError(
            f"partition exhausted at {cumulative_text_tokens:,} text tokens, below target "
            f"{args.target_text_tokens:,}"
        )

    write_final_manifest(
        megatron_root,
        data_root,
        manifest_dir,
        processed_dir,
        args.target_text_tokens,
        selected,
        tokenizer_dir,
    )
    print(f"Selected shards: {len(selected):,}")
    print(f"Text tokens: {cumulative_text_tokens:,}")
    print(f"Megatron data arguments: {processed_dir / 'data_args.txt'}")
    print(f"Dataset manifest: {manifest_dir / 'dataset_manifest.json'}")
    print("NEMOTRON-CC HIGH/ACTUAL DATA PREPARATION: PASS")


if __name__ == "__main__":
    main()
