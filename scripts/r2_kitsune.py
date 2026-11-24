# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Official AfterImage statistics on a disclosed common five-packet-per-edge cohort."""

import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[k] = "1"
import sys, time, types, hashlib, subprocess, gc
from pathlib import Path
import numpy as np
import pandas as pd
import r2_pipeline as p
from r2_addenda import fitcell, OUT

vendor = Path(__file__).resolve().parent.parent / "vendor/Kitsune-py"
# Pure-Python upstream statistics; skip optional Cython hook, not feature logic.
sys.path.insert(0, str(vendor))
source = (
    (vendor / "netStat.py")
    .read_text()
    .replace("import pyximport", "")
    .replace("pyximport.install()", "")
)
afterimage = types.ModuleType("AfterImage")
exec(
    compile(
        (vendor / "AfterImage.py").read_text().replace("np.Inf", "np.inf"),
        str(vendor / "AfterImage.py"),
        "exec",
    ),
    afterimage.__dict__,
)
sys.modules["AfterImage"] = afterimage
module = types.ModuleType("official_netstat")
exec(compile(source, str(vendor / "netStat.py"), "exec"), module.__dict__)
rec, content = p.load_records("5G-NIDD")
original = pd.read_pickle(p.WORK / "cache/5G-NIDD/aggregate_v2_sparse.pkl")
edge = original["record_edge"]
original_edges = original["edges"]
eligible_edges = original_edges.loc[original["keep"]]
# Metadata-only bounded cohort, fixed before any official-extractor score.
chosen = np.concatenate(
    [
        g.sample(n=min(64, len(g)), random_state=2026).index.to_numpy()
        for _, g in eligible_edges.groupby(
            ["capture", "label"], observed=True, sort=True
        )
    ]
)
eligible = np.isin(edge, chosen)
ordinal = pd.Series(edge).groupby(edge, sort=False).cumcount().to_numpy()
select = eligible & (ordinal < 5)
sub = rec.loc[select].copy()
del ordinal, rec, original
gc.collect()
macsrc = []
macdst = []
offset = 0
for file in sorted((p.ROOT / "features/payload_v2_4class/5G-NIDD").glob("*.npy")):
    raw = np.load(file, mmap_mode="r")
    mask = select[offset : offset + len(raw)]
    macsrc.extend(raw["src_mac"][mask].astype(str).tolist())
    macdst.extend(raw["dst_mac"][mask].astype(str).tolist())
    offset += len(raw)
sub["srcmac"] = macsrc
sub["dstmac"] = macdst
del macsrc, macdst, select
a = p.aggregate(sub, content, "5G-NIDD")
e = a["edges"]
sub["edge"] = a["record_edge"]
provenance = dict(
    commit=(vendor / "COMMIT").read_text().strip(),
    url="https://github.com/ymirsky/Kitsune-py",
    source_hashes={
        n: hashlib.sha256((vendor / n).read_bytes()).hexdigest()
        for n in ("netStat.py", "AfterImage.py", "LICENSE")
    },
    compatibility="Optional pyximport hook omitted; np.Inf spelling replaced by np.inf for NumPy 2. Official pure-Python netStat and AfterImage formulas unchanged.",
    records=len(sub),
    edges=len(e),
    original_edge_indices=chosen.tolist(),
    selection="at most 64 clean eligible edges per capture/majority-class, random_state=2026; first five released records of each; fixed before any official-extractor score. Larger-cohort attempt terminated for computational cost before scores.",
    representation="100 official AfterImage statistics, arithmetic mean per edge",
    state="reset for every capture and train/test partition, chronological order within each; no label supplied to extractor",
    classifier="ExtraTrees; same split, budget and labeled indices for both feature families",
    scope="bounded, class-covered comparative extraction, not whole-capture Kitsune/KitNET detection or original class prevalence",
)
p.savejson(OUT / "kitsune_provenance.json", provenance)
for split, pool, test in p.splits(a, "5G-NIDD"):
    target = OUT / ("kitsune_features_" + split + ".npy")
    if (
        split == "holdout_BS2"
        and not target.exists()
        and (OUT / "kitsune_features_holdout_BS1.npy").exists()
    ):
        # Both directions contain whole captures; state resets per capture, so the
        # matrices are identical even though supervised training/test roles reverse.
        os.link(OUT / "kitsune_features_holdout_BS1.npy", target)
    if target.exists():
        X = np.load(target)
    else:
        X = np.zeros((len(e), 100))
        counts = np.zeros(len(e), int)
        start = time.perf_counter()
        for members in (pool, test):
            selected = sub[sub.edge.isin(members)]
            for capture, group in selected.groupby("capture", observed=True):
                stat = module.netStat(HostLimit=100000, HostSimplexLimit=1000)
                for row in group.sort_values(
                    ["ts", "record_id"], kind="stable"
                ).itertuples():
                    # Official extractor uses ports for TCP/UDP and protocol names otherwise.
                    sp = (
                        str(int(row.sport))
                        if row.proto in (6, 17)
                        else "icmp"
                        if row.proto == 1
                        else ""
                    )
                    dp = (
                        str(int(row.port))
                        if row.proto in (6, 17)
                        else "icmp"
                        if row.proto == 1
                        else ""
                    )
                    x = stat.updateGetStats(
                        0,
                        row.srcmac,
                        row.dstmac,
                        str(row.src),
                        sp,
                        str(row.dst),
                        dp,
                        float(row.size),
                        float(row.ts),
                    )
                    X[row.edge] += x
                    counts[row.edge] += 1
        assert np.all(counts[np.r_[pool, test]] == 5)
        X /= np.maximum(counts[:, None], 1)
        assert np.isfinite(X).all()
        np.save(target, X)
        p.savejson(
            OUT / ("kitsune_extraction_" + split + ".json"),
            dict(
                seconds=time.perf_counter() - start,
                records=int(counts.sum()),
                feature_count=100,
            ),
        )
    for budget in (1, 3, 5):
        for seed in p.SEEDS:
            fitcell(
                a,
                pool,
                test,
                "5G-NIDD",
                "kitsune_common_" + split,
                "AfterImage-official",
                X,
                budget,
                seed,
            )
            fitcell(
                a,
                pool,
                test,
                "5G-NIDD",
                "kitsune_common_" + split,
                "TCV6",
                None,
                budget,
                seed,
            )
    print("Kitsune completed", split, flush=True)
p.savejson(OUT / "completion_kitsune.json", provenance)
