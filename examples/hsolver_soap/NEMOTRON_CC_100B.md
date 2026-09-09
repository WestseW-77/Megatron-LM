# Reproducing the 100B Nemotron-CC corpus

The formal Hsolver SOAP experiment uses a deterministic sample from the public
high-quality real-text partition of Nemotron-CC:

```text
quality=high/kind=actual/kind2=actual
```

The Nemotron-CC paper reports approximately 553 billion tokens in this
partition. The experiment preparation target is 100,000,000,000 text tokens
under the pinned experiment tokenizer. The target describes the available
corpus, not the number of tokens consumed by a training run.

## Pinned inputs

| Input | Pin |
| --- | --- |
| Common Crawl object list | `data-jsonl.paths.gz` |
| Object-list SHA256 | `d8201c1e05b5e9fecef7678457c172f98a19a25f1dd656cdba1f937e7a29e68e` |
| Partition | `quality=high/kind=actual/kind2=actual` |
| Partition files | 2,755 |
| Common Crawl snapshots | 99 |
| Sampling namespace | `hsolver-nemotron-100b-v1` |
| Sampling-order SHA256 | `e693749197c2899ebabb27c4f44024d823425c69e372031d409c986ac64d94d4` |
| Tokenizer | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16@97ab8012882a655dc38df4fee47422aca9caca07` |

The tokenizer derivation is identical to the small-sample experiment described
in [DATA.md](DATA.md): it preserves all 131,072 token IDs, designates `</s>`
(ID 2) as EOD, and appends one EOD token to every nonempty document.

## Sampling rule

The source partition spans 99 Common Crawl snapshots whose shard counts differ.
Selecting the first paths in the upstream list would bias the sample toward
particular crawl dates. The preparation script instead constructs one stable
stratified order:

1. Group paths by `CC-MAIN-YYYY-WW` snapshot.
2. Within each snapshot, sort paths by
   `SHA256("hsolver-nemotron-100b-v1" + NUL + path)`.
3. For one-based rank `r` among `n` paths, assign midpoint `(2r - 1)/(2n)`.
4. Sort all paths by midpoint, hash, and path.
5. Tokenize complete shards in that order until cumulative text tokens are at
   least 100,000,000,000.

This makes every prefix of the order approximately proportional to each
snapshot's shard count. It never truncates a document. The final corpus will
slightly exceed 100B because the stopping unit is a complete source shard.

The 100B estimate corresponds to roughly 499 of the 2,755 source shards. The
actual number is determined by the pinned tokenizer rather than by an assumed
average number of tokens per shard.

## Storage layout

For a data root such as `/data1/ll/HPC/data/soap-8b-formal`, the script writes:

```text
sources/nemotron-cc/
  data-jsonl.paths.gz
  high-actual/*.jsonl.zstd
tokenizers/
  nemotron-3-nano-30b-a3b-base/
  nemotron-3-nano-30b-a3b-base-pretrain-eod2/
manifests/nemotron-cc-high-actual-100b/
  sampling_plan.json
  sampling_order.jsonl
  selected_paths.txt
  selected_shards.jsonl
  data_prefixes.relative.txt
  dataset_manifest.json
processed/nemotron-cc-high-actual-100b-eod2/
  data_args.txt
  metadata/*.json
  shards/*_text_document.bin
  shards/*_text_document.idx
```

Each source shard becomes an independent Megatron indexed dataset. This avoids
creating a second approximately 400 GB copy merely to merge all token files.
`data_args.txt` contains absolute indexed-dataset prefixes, one per line;
Megatron infers blend weights from their lengths.

Expected disk usage is approximately 175-190 GB for compressed source shards
and approximately 400 GB for `int32` token data, plus indexes and temporary
space. Reserve 1-1.5 TiB. When compressed shards are removed after successful
indexing, long-term use is lower, but reconstructing a damaged index requires a
new download.

## Inspect the plan first

Data preparation is CPU and storage work; it does not require a GPU allocation.
The wrapper clears `CUDA_VISIBLE_DEVICES` so it cannot occupy one of the shared
training GPUs. From the Megatron-LM repository root:

```bash
NEMOTRON_CC_PLAN_ONLY=1 \
bash examples/hsolver_soap/data/prepare_nemotron_cc_high_actual.sh \
  /data1/ll/HPC/data/soap-8b-formal
```

This downloads only the small official object list, verifies its SHA256, and
writes the complete sampling order. It does not download dataset shards or load
the tokenizer.

## Prepare the corpus

Choose the worker count from the CPU cores allocated to this process. For
example, with 32 allocated CPU cores:

```bash
NEMOTRON_CC_WORKERS=32 \
bash examples/hsolver_soap/data/prepare_nemotron_cc_high_actual.sh \
  /data1/ll/HPC/data/soap-8b-formal
```

The default is eight workers. Each worker loads one tokenizer instance. The
script performs these operations for each selected shard:

1. Resume or start its HTTPS download into a `.part` file.
2. Verify transfer length and calculate source SHA256.
3. Stream Zstandard decompression without materializing uncompressed JSONL.
4. Tokenize records in source order using a multiprocessing pool.
5. Write and validate an atomic Megatron `.bin/.idx` pair.
6. Calculate output SHA256 and write per-shard completion metadata.

Completed shards are reused after interruption. Incomplete indexed `.building`
files are regenerated, and partial downloads resume. By default, existing
outputs are checked by metadata and file size. To reread and hash all completed
indexed files during an audit, set:

```bash
NEMOTRON_CC_VERIFY_EXISTING_HASHES=1
```

To remove each compressed source shard only after its indexed output and
metadata are complete, set:

```bash
NEMOTRON_CC_REMOVE_COMPRESSED=1
```

The script honors the shell's proxy variables by default. If this specific
Common Crawl endpoint must bypass a configured proxy, use:

```bash
NEMOTRON_CC_IGNORE_PROXY=1
```

This setting affects only the preparation process and does not modify shell or
Conda proxy configuration.

After all source files and tokenizer files are already local, network access can
be prohibited explicitly:

```bash
NEMOTRON_CC_OFFLINE=1
```

## Use with Megatron

Replace a single `--data-path` argument with the generated argument file:

```bash
--data-args-path \
  /data1/ll/HPC/data/soap-8b-formal/processed/nemotron-cc-high-actual-100b-eod2/data_args.txt
```

Continue to provide the chosen `--split`. The Torch-SOAP and Hsolver-SOAP runs
must use the same `data_args.txt`, split, data-cache directory contents, seed,
parallel configuration, batch sizes, sequence length, and training-step count.

For Megatron's usual fixed-length packed samples, the reported training-token
horizon is:

```text
train_iters * global_batch_size * sequence_length
```

It is independent of the 100B source-corpus capacity as long as the selected
training split contains enough samples.

## Publication artifacts

Do not commit the source or generated `.bin/.idx` data. Archive and publish the
preparation script plus these small metadata files:

- `sampling_plan.json`
- `sampling_order.jsonl`
- `selected_paths.txt`
- `selected_shards.jsonl`
- `dataset_manifest.json`

Together they record the complete selected source set, per-object and
per-indexed-shard SHA256 values, exact document and token counts, snapshot
distribution, tokenizer pin, and Megatron commit. The generated
`data_args.txt` is machine-specific because it contains absolute paths and does
not need to be published.

The Common Crawl-hosted source remains governed by the Common Crawl Terms of
Use. Reproducers download it from the original host rather than from this
repository.
