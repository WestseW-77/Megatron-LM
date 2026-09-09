# Reproducing the SOAP 8B pretraining data

This document describes the original small development corpus. The formal
100B high-quality real-text corpus and its deterministic sampling procedure are
documented separately in [NEMOTRON_CC_100B.md](NEMOTRON_CC_100B.md).

This experiment uses one pinned shard from NVIDIA's public
`Nemotron-Pretraining-Dataset-sample` repository. The generated JSONL and
Megatron indexed dataset are local build artifacts and are not distributed
with this repository.

## Why this source and subset

The experiment compares two QR implementations inside the same SOAP training
pipeline. It is therefore useful to keep data preparation simple and remove
mixture weights as an additional experimental variable.

We use only `Nemotron-CC-Translated-Diverse-QA/part_000000.parquet` because it
is public, small enough for an independently reproducible systems experiment,
and approximately balanced across 15 languages. We do not mix subsets,
shuffle rows, or deduplicate text. All 15,441 source rows are retained in their
original order, including 12 repeated text rows.

This data is used for random-initialized next-token pretraining. It is not an
SFT, preference-optimization, or reinforcement-learning dataset.

## License and redistribution

The source repository is governed by the NVIDIA Open Data License Agreement
included in its `LICENSE.md`. At the pinned revision, the terms restrict making
the dataset available to others. Anyone reproducing this experiment must
review and accept the source terms and download the files directly from
NVIDIA. Do not commit or redistribute the downloaded Parquet, derived JSONL,
or generated `.bin/.idx` files with these scripts.

The source README also describes additional model-redistribution conditions
that may apply to models trained on the data. Review the current upstream terms
for the intended use. This document records the experiment configuration and
is not legal advice.

## Pinned inputs

| Input | Repository revision | Selected file |
| --- | --- | --- |
| Dataset | `nvidia/Nemotron-Pretraining-Dataset-sample@3ad096e6394e487bb4f778733300da85275bb449` | `Nemotron-CC-Translated-Diverse-QA/part_000000.parquet` |
| Tokenizer | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16@97ab8012882a655dc38df4fee47422aca9caca07` | tokenizer/config files only; no model weights |

The tokenizer vocabulary has 131,072 entries. The preparation script changes
the tokenizer's EOS designation from the chat terminator `<|im_end|>` (ID 11)
to the base-model document terminator `</s>` (ID 2). It does not change the
vocabulary or token IDs. Megatron's `--append-eod` then appends ID 2 once to
every document.

## Software used for the reference artifacts

- Python 3.12.13
- PyTorch 2.12.1+cu132
- NumPy 2.5.1
- Transformers 4.55.4
- Tokenizers 0.21.4
- huggingface-hub 0.36.2
- PyArrow 25.0.1
- Megatron Core 0.20.0+5714a206e

The preparation itself is CPU-only and does not require a GPU allocation.

## Build

From the Megatron-LM repository root, after activating an environment with the
packages above:

```bash
bash examples/hsolver_soap/data/prepare_nemotron_sample.sh \
  "$HOME/HPC/data/soap-8b"
```

The optional `SOAP_DATA_WORKERS` environment variable controls Megatron
preprocessing workers and defaults to 8. To validate already downloaded local
artifacts without contacting Hugging Face, set `SOAP_OFFLINE=1`.

```bash
SOAP_OFFLINE=1 \
bash examples/hsolver_soap/data/prepare_nemotron_sample.sh \
  "$HOME/HPC/data/soap-8b"
```

The script is fail-safe on existing artifacts: it reuses files only when their
hashes match and refuses to overwrite mismatched outputs.

## Deterministic transformations

1. Download the pinned Parquet shard, source README/license, and four tokenizer
   configuration files.
2. Derive the pretraining tokenizer by setting EOS to `</s>` (ID 2), without
   changing its vocabulary.
3. Convert Parquet rows to compact UTF-8 JSONL containing `id`, `language`, and
   `text`, in that field order. Preserve source order and LF endings; do not
   shuffle or deduplicate.
4. Invoke `tools/preprocess_data.py` with the Hugging Face tokenizer,
   `--append-eod`, one partition, and sequential sample order.
5. Read the generated indexed dataset and validate document boundaries, token
   counts, EOD placement, vocabulary bounds, file sizes, and hashes.

## Expected statistics

| Statistic | Expected value |
| --- | ---: |
| Source/JSONL documents | 15,441 |
| Unique IDs | 15,441 |
| Repeated text rows retained | 12 |
| Text tokens | 14,943,607 |
| Appended EOD tokens | 15,441 |
| Total tokens | 14,959,048 |
| Minimum document length | 11 |
| Maximum document length | 4,130 |
| Minimum token ID | 2 |
| Maximum token ID | 131,070 |

## Reference SHA256 values

| Artifact | SHA256 |
| --- | --- |
| Source Parquet | `835851141069310879d95d49363d1c96e280a552ac4d4b53a6dc83c30e9e34a2` |
| Derived JSONL | `3eccdead5ea3b707be399b6ac3d693b9f1d70d22205e180aa8d163f687a550a1` |
| JSONL conversion manifest | `c67262255c5e042648a268d6c6320331ceac553d49a3b4981ecc3f9bb5bf9d2c` |
| Derived `config.json` | `c78db134b3aecd82042b9a573bd0d71acabfee3f1b4d082fe78d1c1d317cebfb` |
| Derived `derivation.json` | `56473d1c61883e35d4ccaa318851a66214db110ce3b55e2fbea3805d5f50bbf4` |
| Derived `special_tokens_map.json` | `afe7d139eb6dc4aff93f89749945dce6fc059c22cbaa394724ae01b8dd40339d` |
| Derived `tokenizer.json` | `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7` |
| Derived `tokenizer_config.json` | `9a4c2659ce205101e66178b891f29d450c394edf3fa3c0944503379f3d0a0416` |
| Megatron `.bin` | `2ac059d4295f3c1e5600f4ab47d44bb22bac9717e655519166686de3f8e0fa3e` |
| Megatron `.idx` | `4df050a7b382da46c20ebd80eb736c6cbad3ab510a5c3d0bc5c6d0fa7dd99e5d` |
