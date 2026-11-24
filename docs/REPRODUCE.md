# Reproduction guide

Run commands from the repository root, using Python 3.12 on Linux/WSL2 for the
full suite. Install `requirements-experiments.txt` in an isolated environment.
Pinned versions match the experiment snapshot; hardware and operating-system
differences can affect timing and some numerical results.

## 1. Set paths

```bash
export IOT_DATA_ROOT="$PWD/data_root"
export FIVEG_RAW_ROOT="/absolute/path/to/5G-NIDD"
export IOT_REVISION_ROOT="$PWD/revision_run"
mkdir -p "$IOT_DATA_ROOT/data/v5_raw/TON_IoT" "$IOT_DATA_ROOT/features/logs" "$IOT_REVISION_ROOT"
```

For the core pipeline on PowerShell:

```powershell
$env:IOT_DATA_ROOT = "$PWD/data_root"
$env:FIVEG_RAW_ROOT = 'D:/datasets/5G-NIDD'
$env:IOT_REVISION_ROOT = "$PWD/revision_run"
New-Item -ItemType Directory -Force $env:IOT_REVISION_ROOT
```

Environment variables are read at process startup. Follow DATASETS.md to place
the TON CSV and generate corrected 5G shards. Never run the old extractor as a
standalone preparation command; only its parser helpers are included.

## 2. Check and prepare

```bash
python scripts/test_r2_pipeline.py
python scripts/smoke_test.py
python scripts/test_release.py
python scripts/r2_pipeline.py --dataset TON-IoT --prepare
python scripts/r2_pipeline.py --dataset 5G-NIDD --prepare
```

Preparation writes `cache/<dataset>/records.pkl`, `content.npy`, and
`aggregate_v2_sparse.pkl`. The packet corpus is memory intensive: the sparse
application matrix avoids dense edge-by-application allocation, but raw records,
multiple rebuilt edge sets and classifiers still require substantial RAM.

## 3. Pilot, then primary suite

```bash
python scripts/r2_pipeline.py --dataset TON-IoT --pilot
python scripts/r2_pipeline.py --dataset 5G-NIDD --pilot
python scripts/r2_pipeline.py --dataset TON-IoT
python scripts/r2_pipeline.py --dataset 5G-NIDD
```

Pilot means one seed, 5% labels, random split and one rotation level; it is not
the published multi-seed experiment. Full primary runs cover 990 model cells
(495 per dataset), including 5% single-component and leave-one-component-out
ablations. Primary outputs go to `results/<dataset>/`; pilots go to
`pilot_results/<dataset>/`.

## 4. Reviewer-driven experiments

```bash
python scripts/r2_addenda.py ton
python scripts/r2_addenda.py fiveg
python scripts/r2_addenda.py windows
python scripts/r2_kitsune.py
python scripts/r2_cost.py
```

| Command | Experiments |
|---|---|
| `ton` | Native Zeek fields, five-record versus five-packet support, training versus corpus reference sensitivity, stacking |
| `fiveg` | Matched Argus baseline, withheld Goldeneye/Slowloris/TCPConnect captures, exact-feature collision audit, stacking |
| `windows` | 1/10/60-second windows, support checks, record coverage, closure delay and missed events |
| `r2_kitsune.py` | Official AfterImage mean statistics versus TCV on the same bounded cohort |
| `r2_cost.py` | Three repetitions at 2,000/5,000/10,000 packets from the specified capture |

All these write `addenda/`. Stacking requires LightGBM and XGBoost. AfterImage
requires the included upstream files and the prepared 5G cache. Cost requires
Scapy, original PCAPNG and Linux `/proc/cpuinfo`. Timing is a warm-process offline
batch measurement, not live device throughput. Keep full machine/run metadata
when comparing it with the paper.

## 5. Inspect outputs

```bash
python scripts/summarize.py "$IOT_REVISION_ROOT" --output "$IOT_REVISION_ROOT/summary.csv"
python -m pip freeze > "$IOT_REVISION_ROOT/pip-freeze.txt"
git rev-parse HEAD > "$IOT_REVISION_ROOT/code-commit.txt"
```

The final command requires a local commit. Primary JSON includes per-class
metrics, confusion matrices, timings, benign-reference support and rotation
coverage. Matching NPZ files contain test/labeled indices and predictions.
`design.json` identifies train/test membership; `environment.json` records
runtime versions and the pipeline hash. Addenda JSON carries experiment-specific
support and provenance; its NPZ files preserve classifications/OOF evidence.
The summary script aggregates primary clean metrics only; use addenda JSON for
the other experiment families. Compare generated results with the paper.

## Reruns and troubleshooting

- **Use a new `IOT_REVISION_ROOT` for any changed input, code or protocol.** Existing
  caches/results are reused. Resume is intended only for interrupted identical
  runs; it is not a complete cache-invalidation system.
- Missing shards: run both preparation steps. Missing TON columns: confirm the
  original network CSV, not a re-encoded mirror or different modality.
- Missing classes: preserve the failure/unavailable result. Do not silently
  change the test set or increase the annotation budget to obtain a score.
- `resource` or `/proc/cpuinfo` unavailable: use Linux/WSL2 for addenda and cost.
- Missing LightGBM runtime on Linux: install the platform's OpenMP runtime.
- Never load pickle caches from an untrusted source. This release contains none.

## Publishing this repository

The local repository is prepared for `https://github.com/apresekal/TCV`.
After reviewing the files and creating the empty GitHub repository, the owner
can commit and push. Dataset folders, local environments and run outputs are
ignored. The preparation process does not publish or push anything.
