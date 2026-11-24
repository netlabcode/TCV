# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Scientific invariants, independent of manuscript numerical results."""

import numpy as np
import pandas as pd
from r2_pipeline import aggregate, reference, features, sample_labels


def main():
    # Two edges, deliberately very different payload/size/timing populations.
    d = pd.DataFrame(
        {
            "capture": ["c"] * 20,
            "proto": [6] * 20,
            "src": ["s1"] * 10 + ["s2"] * 10,
            "dst": ["d"] * 20,
            "port": [80] * 20,
            "size": [100.0] * 10 + [900.0] * 10,
            "packet_weight": [1.0] * 20,
            "duration": [0.0] * 20,
            "ttl": [64] * 20,
            "flags": [2] * 20,
            "ts": list(np.arange(10.0)) + list(np.arange(10.0) * 10),
            "iat": [999.0] * 20,
            "l7": ["http"] * 20,
            "label": ["Normal"] * 10 + ["Flooding"] * 10,
            "record_id": np.arange(20),
        }
    )
    x = np.concatenate([np.zeros((10, 4)), np.full((10, 4), 200.0)])
    a = aggregate(d, x, "5G-NIDD")
    ref = reference(a, np.array([0]))
    f = features(a, ref)
    assert np.allclose(ref["content"], 0) and np.isclose(ref["median"], 1)
    assert np.isclose(f[0, 0], 0) and np.isclose(f[1, 0], 200)
    altered = dict(ref)
    altered["l7"] = {"http": 0.75, "unseen_in_test": 0.25}
    expected = 0.5 * (
        np.log(1 / 0.875) + 0.75 * np.log(0.75 / 0.875) + 0.25 * np.log(2)
    )
    assert np.allclose(features(a, altered)[:, 4], expected, atol=1e-12), (
        "sparse JS lost baseline mass outside test support"
    )
    assert np.isclose(a["edges"].iat_median.iloc[1], 10), (
        "cached historical IAT must not be reused"
    )
    changed = x.copy()
    changed[10:] = 255
    b = aggregate(d, changed, "5G-NIDD")
    ref2 = reference(b, np.array([0]))
    assert np.array_equal(ref["content"], ref2["content"]), (
        "test payload contaminated training reference"
    )
    # Rotate every three records: all fragments below the five-record threshold.
    rot = aggregate(d, x, "5G-NIDD", rotation=3)
    assert rot["edges"].n_records.sum() == 20 and rot["keep"].sum() == 0
    assert len(rot["edges"]) == 8
    win = aggregate(d, x, "5G-NIDD", window=5)
    assert len(win["edges"]) > 2 and win["edges"].n_records.sum() == 20
    # Flow support counts records, not the number of packets summarized by them.
    flow = d.iloc[:4].copy()
    flow.packet_weight = 100
    f = aggregate(flow, x, "TON-IoT")
    assert f["keep"].sum() == 0
    labels = np.array(
        ["Normal"] * 200 + ["Scanning"] * 580 + ["Exploit"] * 43 + ["Flooding"] * 7
    )
    selected = sample_labels(np.arange(len(labels)), labels, 1, 42)
    assert len(selected) == 8 and len(set(labels[selected])) == 4
    rng = np.random.default_rng(2026)
    groups = rng.integers(0, 30, 1000)
    bins = rng.integers(0, 32, 1000)
    old = np.zeros((30, 32), np.int64)
    np.add.at(old, (groups, bins), 1)
    fast = np.bincount(groups * 32 + bins, minlength=30 * 32).reshape(30, 32)
    assert np.array_equal(old, fast), "fast count aggregation changed statistics"
    print(
        "PASS: train-only reference, IAT recomputation, rotation conservation, windowing, record support"
    )


if __name__ == "__main__":
    main()
