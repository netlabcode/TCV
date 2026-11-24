# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Exercise aggregation, label sampling, fitting and scoring without real data.

Synthetic observations test execution and scientific invariants, not accuracy
on either public dataset. No files or dataset caches are created.
"""

import numpy as np
import pandas as pd
import r2_pipeline as p


def main():
    rng = np.random.default_rng(2026)
    records, content = [], []
    classes = ["Normal", "Flooding", "Scanning", "Slow-rate"]
    for edge in range(400):
        label = classes[edge % 4]
        for ordinal in range(6):
            records.append(
                dict(
                    capture="synthetic_BS1_capture"
                    if edge < 200
                    else "synthetic_BS2_capture",
                    proto=6,
                    src=f"source-{edge}",
                    dst="destination",
                    port=80,
                    size=float(100 + 100 * (edge % 4) + rng.uniform(0, 10)),
                    packet_weight=1.0,
                    duration=0.0,
                    ttl=64,
                    flags=2,
                    ts=float(ordinal * (1 + edge % 4)),
                    iat=0.0,
                    l7="http",
                    label=label,
                    record_id=len(records),
                )
            )
            content.append(rng.uniform(edge % 4, 1 + edge % 4, 64))
    records = pd.DataFrame(records)
    a = p.aggregate(records, np.array(content), "5G-NIDD")
    assert a["keep"].sum() == 400
    labels = a["edges"].label.to_numpy(str)
    for name, train, test in p.splits(a, "5G-NIDD"):
        assert not set(train).intersection(test)
        labeled = p.sample_labels(train, labels, 5, 42)
        assert len(labeled) == int(len(train) * 0.05)
        assert set(labels[labeled]) == set(classes)
        ref = p.reference(a, labeled)
        X = p.features(a, ref)
        assert X.shape == (400, 6) and np.isfinite(X).all()
        clf = p.model("ExtraTrees", 42).fit(X[labeled], labels[labeled])
        result = p.score(labels[test], clf.predict(X[test]), sorted(classes))
        assert sum(map(sum, result["confusion_matrix"])) == len(test)
        assert 0 <= result["macro_f1"] <= 1
        print(
            f"PASS synthetic fit: {name}; {len(labeled)} labels; {len(test)} test edges"
        )
    try:
        p.aggregate(records, np.array(content), "TON-IoT", window=10)
    except ValueError:
        pass
    else:
        raise AssertionError("TON-IoT windowing must reject absent timestamps")
    print("PASS: synthetic end-to-end workflow (not manuscript performance)")


if __name__ == "__main__":
    main()
