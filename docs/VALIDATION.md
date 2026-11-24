# Release verification — 24 November 2026

Checked locally on Windows with Python 3.12 and the pinned NumPy 2.3.5,
pandas 2.3.3, SciPy 1.17.1 and scikit-learn 1.8.0 dependencies.

| Check | Outcome |
|---|---|
| `python scripts/test_r2_pipeline.py` | Passed: train-only reference, sparse JS union support, IAT recomputation, rotation conservation, windows, five-record support and class-preserving exact annotation budgets |
| `python scripts/smoke_test.py` | Passed: 400 synthetic edges, three split types, annotation sampling, ExtraTrees fitting and metric accounting; TON window rejection |
| `python scripts/test_release.py` | Passed: retained packet parser, payload padding, five-seed Student-t aggregation and SHA-256 input manifest |
| `python scripts/r2_pipeline.py --help` | Passed |
| `python -m compileall -q scripts` | Passed |
| Ruff undefined-name check (`F821`) | Passed |
| `python -m pip check` | Passed |
| `cffconvert --validate` | Passed: CITATION.cff schema 1.2.0 |

The synthetic Scapy check emitted a Windows libpcap-provider warning; it uses a
constructed packet and passed without live capture. The full experiment suite
targets Linux/WSL2. Addenda, AfterImage extraction, raw-data preparation and
offline timing have **not** been rerun on the complete datasets for this release.
Experimental result tables are provided in the paper and are not distributed here.

The code was formatted for readability. The packet helper was reduced to the
five functions actually needed; unrelated dataset and extraction workflows were
removed. Runtime guards reject missing captures/CSVs/shards instead of silently
producing incomplete preparation outputs. These guards do not change the
complete-input scientific protocol.

Dataset landing pages were verified; bulk dataset access/download was not
completed. See DATASETS.md for access details. No manuscript, reviewer response,
raw record, prediction archive, credential or private absolute path is included
in the intended Git file set. The local Python environment is ignored by Git.

The repository uses branch `main` and remote
`https://github.com/apresekal/TCV.git`. The checks above describe local release
verification, independently of repository publication.
