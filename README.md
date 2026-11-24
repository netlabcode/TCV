# TCV — Traffic Characterization Vectors

Core reproducibility code by **Alfan Presekal (`apresekal`)** for:

> Alfan Presekal and Ruki Harwahyu, *Label-Efficient and Leakage-Aware Intrusion
> Detection for IoT Edge Monitoring Using Traffic Characterization Vectors*.
> Manuscript under revision for IEEE Access (2026).

Intended repository: https://github.com/apresekal/TCV. This local release does
not imply that the GitHub repository has already been published.

TCV represents a directed traffic edge with six statistics: content deviation,
timing deviation, size/timing dispersion, packet-size distribution deviation,
application-distribution divergence, and mean-size magnitude. Reference
distributions are learned only from annotated benign training edges.

## Scope

This focused release contains the corrected 5G-NIDD and TON-IoT experiments
used in the November 2026 manuscript: core comparisons, ablations, identity
rotation, native feature baselines, capture/tool holdout, stacking, windowing,
AfterImage comparison, and offline processing cost. It includes the required
packet preparation helpers. Experimental results are provided in the paper.

Historical exploratory experiments, earlier meta-learning implementations,
manuscript/response production scripts, raw datasets, trained models, and saved
per-record predictions are outside this release. Thus it reproduces the
corrected experimental suite, not every historical result retained in the paper.

## Quick start (no dataset required)

Python **3.12** is the reference interpreter. Linux or WSL2 is required for the
complete suite (the cost/addenda scripts use Linux resource accounting).
The core pipeline and synthetic checks also run on Windows.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_r2_pipeline.py
python scripts/smoke_test.py
```

PowerShell activation: `.\.venv\Scripts\Activate.ps1`.
The smoke test is synthetic and does not measure manuscript accuracy.

## Reproduce the manuscript experiments

1. Install the remaining dependencies: `python -m pip install -r requirements-experiments.txt`.
2. Obtain the datasets from their official providers and prepare them using
   [Dataset sources and preparation](docs/DATASETS.md).
3. Follow the exact commands in [Reproduction guide](docs/REPRODUCE.md).
4. Read [Protocol and implementation](docs/PROTOCOL.md) before interpreting scores.
5. Compare your generated results with the corresponding experiments in the paper.

| Path | Purpose |
|---|---|
| `scripts/r2_pipeline.py` | Edge aggregation, six TCV features, train-only references, split/annotation sampling, main experiments and rotation |
| `scripts/r2_addenda.py` | Native features, sensitivity, withheld captures, OOF stacking and time windows |
| `scripts/r2_kitsune.py` | Official AfterImage statistics on a matched bounded cohort |
| `scripts/r2_cost.py` | Raw-PCAP-to-prediction offline batch timing |
| `scripts/legacy_preparation/` | Only the required 5G-NIDD extraction and label correction helpers |
| `scripts/summarize.py` | Aggregate primary run metrics across seeds with 95% Student-t intervals |
| `scripts/input_manifest.py` | Record input SHA-256 hashes before a run |
| `vendor/Kitsune-py/` | Two unchanged upstream modules, MIT license and pinned commit |

## Citation and license

The original TCV code and documentation are licensed under the
**Apache License, Version 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for the license and attribution to Alfan Presekal (`apresekal`).

If you use TCV in your research, please cite the related paper and acknowledge
this repository using [CITATION.cff](CITATION.cff) or [BibTeX](CITATION.bib).
Paper citation is requested as scholarly attribution, not an additional license
condition. The paper citation is provisional; no DOI or accepted-publication
status is asserted. Update the citation after publication. Third-party
AfterImage code retains its original MIT license; see
[the upstream MIT license](vendor/Kitsune-py/LICENSE).

Dataset licenses are separate. Official sources are
[5G-NIDD / NetsLab](https://netslab.ucd-cs.ie/netslab-datasets/5g-nidd-dataset/)
and [TON-IoT / UNSW](https://research.unsw.edu.au/projects/toniot-datasets).

## Reproducibility status

The implementation is curated from the manuscript experiment snapshot.
See `docs/VALIDATION.md` for checks actually performed on this release. A full
dataset rerun is separate from synthetic verification. Use a fresh output
directory whenever inputs, dependencies, parameters or code change.
