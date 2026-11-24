# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Summarize primary clean results across seeds, keeping each split separate.

Usage: python scripts/summarize.py revision_run --output revision_run/summary.csv
Missing seeds are reported by n_seeds; intervals are absent for a single seed.
This utility does not combine holdout directions or include addenda/pilot results.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def summarize(root):
    rows = []
    for path in sorted((root / "results").glob("*/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "clean" not in result:
            continue
        rows.append(
            {
                **{
                    k: result[k]
                    for k in [
                        "dataset",
                        "split",
                        "budget",
                        "representation",
                        "learner",
                        "seed",
                    ]
                },
                "macro_f1": result["clean"]["macro_f1"],
                "accuracy": result["clean"]["accuracy"],
            }
        )
    if not rows:
        raise ValueError("No primary result JSON files found under results/<dataset>/")
    frame = pd.DataFrame(rows)
    keys = ["dataset", "split", "budget", "representation", "learner"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate seed within an experiment cell")
    output = []
    for group, part in frame.groupby(keys, sort=True):
        row = dict(zip(keys, group))
        row["n_seeds"] = len(part)
        row["seeds"] = ",".join(map(str, sorted(part.seed)))
        for metric in ["macro_f1", "accuracy"]:
            values = part[metric].to_numpy(float)
            row[metric + "_mean"] = values.mean()
            row[metric + "_ci95_halfwidth"] = (
                (
                    t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1)
                    / np.sqrt(len(values))
                )
                if len(values) > 1
                else np.nan
            )
        output.append(row)
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        frame = summarize(args.run_root)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} experiment groups to {args.output}")


if __name__ == "__main__":
    main()
