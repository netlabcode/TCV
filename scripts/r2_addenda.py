# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Frozen second-round addenda. No test-score-driven selection; single thread."""

import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[k] = "1"
import sys, json, time, gc, re, socket, resource, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from scipy.stats import spearmanr
import r2_pipeline as p

OUT = p.WORK / "addenda"
OUT.mkdir(parents=True, exist_ok=True)


def emit(name, obj):
    p.savejson(OUT / (name + ".json"), obj)


def fitcell(a, pool, test, ds, tag, rep, X, budget, seed, reference_mode="training"):
    name = f"{ds}_{tag}_{rep}_b{budget}_s{seed}"
    if (OUT / (name + ".json")).exists():
        return
    y = a["edges"].label.to_numpy(str)
    lab = p.sample_labels(pool, y, budget, seed)
    if X is None:
        ref = p.reference(
            a, lab if reference_mode == "training" else np.flatnonzero(a["keep"])
        )
        X = p.features(a, ref)
    clf = p.model("ExtraTrees", seed)
    clf.fit(X[lab], y[lab])
    pred = clf.predict(X[test])
    emit(
        name,
        dict(
            kind="classification",
            dataset=ds,
            experiment=tag,
            representation=rep,
            budget=budget,
            seed=seed,
            n_labeled=len(lab),
            n_train=len(pool),
            n_test=len(test),
            reference_mode=reference_mode,
            **p.score(y[test], pred, sorted(set(y[pool]))),
        ),
    )
    np.savez_compressed(
        OUT / (name + ".npz"),
        train=pool,
        test=test,
        labeled=lab,
        true=y[test],
        pred=pred,
    )


def native_ton(a):
    d = pd.read_csv(
        p.ROOT / "data/v5_raw/TON_IoT/train_test_network.csv", low_memory=False
    )
    # Numeric fields supplied by the dataset; no endpoint, label, or type columns.
    excluded = {"src_ip", "dst_ip", "src_port", "dst_port", "label", "type", "ts"}
    columns = [
        c for c in d.select_dtypes(include="number").columns if c not in excluded
    ]
    d = d[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    d["edge"] = a["record_edge"]
    X = d.groupby("edge", sort=True).mean().reindex(a["edges"].index).to_numpy()
    emit(
        "TON-IoT_native_fields",
        dict(
            columns=columns,
            aggregation="arithmetic mean per identical edge",
            excluded=sorted(excluded),
        ),
    )
    return X


def ton():
    rec, content = p.load_records("TON-IoT")
    a = p.aggregate(rec, content, "TON-IoT")
    X = native_ton(a)
    for split, pool, test in p.splits(a, "TON-IoT"):
        for b in (1, 3, 5):
            for seed in p.SEEDS:
                fitcell(
                    a,
                    pool,
                    test,
                    "TON-IoT",
                    "native_" + split,
                    "Zeek-native",
                    X,
                    b,
                    seed,
                )
    original = a["keep"].copy()
    for rule in ("records5", "packets5"):
        a["keep"] = (
            (a["edges"].n_records.to_numpy() >= 5)
            if rule == "records5"
            else (a["edges"].n_packets.to_numpy() >= 5)
        )
        for split, pool, test in p.splits(a, "TON-IoT"):
            for ref in ("training", "corpus"):
                for seed in p.SEEDS:
                    fitcell(
                        a,
                        pool,
                        test,
                        "TON-IoT",
                        f"sensitivity_{rule}_{split}",
                        f"TCV6-{ref}",
                        None,
                        5,
                        seed,
                        ref,
                    )
    a["keep"] = original
    stems(a, "TON-IoT")


def stems(a, ds):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    y0 = a["edges"].label.to_numpy(str)
    enc = LabelEncoder().fit(y0)
    y = enc.transform(y0)
    for split, pool, test in p.splits(a, ds):
        if split not in ("random_fixed20", "holdout_BS2"):
            continue
        for seed in p.SEEDS:
            tag = f"{ds}_stack_{split}_s{seed}"
            if (OUT / (tag + ".json")).exists():
                continue
            lab = p.sample_labels(pool, y0, 5, seed)
            counts = np.bincount(y[lab])
            k = min(3, int(counts.min()))
            if k < 2:
                emit(
                    tag,
                    dict(
                        kind="stack_unavailable",
                        dataset=ds,
                        split=split,
                        seed=seed,
                        n_labeled=len(lab),
                        support=counts.tolist(),
                        reason="fewer than two annotations in a class; valid stratified OOF stacking unavailable",
                    ),
                )
                continue
            estimators = [
                (
                    "RF",
                    RandomForestClassifier(
                        n_estimators=120, n_jobs=1, random_state=seed
                    ),
                ),
                ("ET", p.model("ExtraTrees", seed)),
                (
                    "LGB",
                    LGBMClassifier(
                        n_estimators=100, n_jobs=1, verbosity=-1, random_state=seed
                    ),
                ),
                (
                    "XGB",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        n_jobs=1,
                        random_state=seed,
                        eval_metric="mlogloss",
                    ),
                ),
                (
                    "HGB",
                    HistGradientBoostingClassifier(max_iter=100, random_state=seed),
                ),
            ]
            folds = list(
                StratifiedKFold(k, shuffle=True, random_state=seed).split(lab, y[lab])
            )
            oof = np.empty((len(lab), len(estimators) * 4))
            xp = np.empty((len(lab), 6))
            xtest = []
            constituents = {}
            for j, (name, est) in enumerate(estimators):
                for tr, va in folds:
                    ref = p.reference(a, lab[tr])
                    X = p.features(a, ref)
                    fit = clone(est).fit(X[lab[tr]], y[lab[tr]])
                    oof[va, j * 4 : (j + 1) * 4] = fit.predict_proba(X[lab[va]])
                    xp[va] = X[lab[va]]
                X = p.features(a, p.reference(a, lab))
                fit = clone(est).fit(X[lab], y[lab])
                pr = fit.predict_proba(X[test])
                xtest.append(pr)
                constituents[name] = p.score(
                    y0[test], enc.inverse_transform(pr.argmax(1)), enc.classes_.tolist()
                )
            ztest = np.concatenate(xtest, axis=1)
            scaler = StandardScaler().fit(xp)
            extra = scaler.transform(xp)
            extra_test = scaler.transform(X[test])
            variants = {}
            saved = {}
            for removed in [-1, *range(5)]:
                cols = [c for c in range(20) if removed < 0 or c // 4 != removed]
                meta = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(
                    np.column_stack([oof[:, cols], extra]), y[lab]
                )
                yp = enc.inverse_transform(
                    meta.predict(np.column_stack([ztest[:, cols], extra_test]))
                )
                name = "STEM" if removed < 0 else "drop_" + estimators[removed][0]
                variants[name] = p.score(y0[test], yp, enc.classes_.tolist())
                saved[name] = yp
            errors = np.column_stack(
                [oof[:, j * 4 : (j + 1) * 4].argmax(1) != y[lab] for j in range(5)]
            ).astype(float)
            corr = np.corrcoef(errors, rowvar=False)
            corr = np.where(np.isfinite(corr), corr, 0)
            emit(
                tag,
                dict(
                    kind="stack",
                    dataset=ds,
                    split=split,
                    seed=seed,
                    n_labeled=len(lab),
                    support=counts.tolist(),
                    folds=k,
                    base_names=[n for n, _ in estimators],
                    error_correlation=corr.tolist(),
                    constituents=constituents,
                    variants=variants,
                    reference="refit inside every OOF training fold; final fit on outer labeled set",
                ),
            )
            np.savez_compressed(
                OUT / (tag + ".npz"),
                labeled=lab,
                test=test,
                true=y0[test],
                oof=oof,
                oof_true=y[lab],
                **saved,
            )
            print("stack", tag, flush=True)


ARGUS_FIELDS = [
    "Dur",
    "Mean",
    "StdDev",
    "Sum",
    "Min",
    "Max",
    "sTtl",
    "dTtl",
    "sHops",
    "dHops",
    "TotPkts",
    "SrcPkts",
    "DstPkts",
    "TotBytes",
    "SrcBytes",
    "DstBytes",
    "sMeanPktSz",
    "dMeanPktSz",
    "Load",
    "SrcLoad",
    "DstLoad",
    "Loss",
    "pLoss",
    "Rate",
    "SrcRate",
    "DstRate",
    "TcpRtt",
    "SynAck",
    "AckDat",
]


def port(v):
    s = str(v).strip().lower()
    try:
        return int(s, 16) if s.startswith("0x") else int(float(s))
    except ValueError:
        try:
            return socket.getservbyname(s)
        except OSError:
            return -1


def argus(a):
    target = OUT / "argus_edge_means.pkl"
    if target.exists():
        return pd.read_pickle(target)
    frames = []
    audit = []
    for f in sorted(
        (Path(os.environ["FIVEG_RAW_ROOT"]) / "csv").glob("BS*_each_attack/*.csv")
    ):
        d = pd.read_csv(f, low_memory=False)
        d.columns = d.columns.str.strip()
        bs = re.search(r"BS([12])", str(f)).group(1)
        attack = re.sub(r"[12]$", "", f.stem).lower()
        e = a["edges"]
        cap = next(
            (
                c
                for c in e.capture.astype(str).unique()
                if c.lower() == f"5gnidd_{attack}_bs{bs}_nogtp"
            ),
            None,
        )
        if cap is None:
            continue
        d["capture"] = cap
        d["proto"] = (
            d.Proto.astype(str)
            .str.strip()
            .map({"tcp": 6, "udp": 17, "icmp": 1})
            .fillna(0)
            .astype(int)
        )
        d["src"] = d.SrcAddr.astype(str).str.strip()
        d["dst"] = d.DstAddr.astype(str).str.strip()
        d["port"] = d.Dport.map(port)
        label = d.Label.astype(str).str.strip().str.lower() != "benign"
        unordered = pd.DataFrame(
            {
                "proto": d.proto,
                "lo": np.minimum(d.src, d.dst),
                "hi": np.maximum(d.src, d.dst),
                "attack": label.astype(int),
            }
        )
        mixed = unordered.groupby(["proto", "lo", "hi"]).attack.agg(
            ["min", "max", "size"]
        )
        audit.append(
            dict(
                capture=cap,
                rows=len(d),
                coarse_keys=len(mixed),
                mixed_keys=int((mixed["min"] != mixed["max"]).sum()),
                rows_in_mixed_keys=int(
                    mixed.loc[mixed["min"] != mixed["max"], "size"].sum()
                ),
                unknown_destination_ports=int((d.port < 0).sum()),
                tools=sorted(d["Attack Tool"].astype(str).unique().tolist()),
            )
        )
        for c in ARGUS_FIELDS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        frames.append(
            d[p.KEY + ARGUS_FIELDS].groupby(p.KEY, observed=True).mean().reset_index()
        )
    means = pd.concat(frames, ignore_index=True)
    joined = a["edges"][p.KEY].merge(means, on=p.KEY, how="left", validate="one_to_one")
    pd.to_pickle(joined, target)
    emit(
        "5G-NIDD_label_join_audit",
        dict(
            captures=audit,
            matched_edges=int(joined[ARGUS_FIELDS].notna().any(axis=1).sum()),
            total_edges=len(joined),
            fields=ARGUS_FIELDS,
            join="directed capture/protocol/source/destination/destination-port; no reverse-flow duplication",
            limitation="Audit of original coarse label join; no timestamp-complete packet relabeling. CSV times are truncated minute/second strings.",
        ),
    )
    return joined


def fiveg():
    a = pd.read_pickle(p.WORK / "cache/5G-NIDD/aggregate_v2_sparse.pkl")
    native = argus(a)
    eligible = a["keep"].copy()
    matched = native[ARGUS_FIELDS].notna().any(axis=1).to_numpy()
    a["keep"] = eligible & matched
    X = native[ARGUS_FIELDS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    for split, pool, test in p.splits(a, "5G-NIDD"):
        for b in (1, 3, 5):
            for seed in p.SEEDS:
                fitcell(
                    a,
                    pool,
                    test,
                    "5G-NIDD",
                    "native_matched_" + split,
                    "Argus-native",
                    X,
                    b,
                    seed,
                )
                fitcell(
                    a,
                    pool,
                    test,
                    "5G-NIDD",
                    "native_matched_" + split,
                    "TCV6",
                    None,
                    b,
                    seed,
                )
    a["keep"] = eligible
    e = a["edges"]
    valid = np.flatnonzero(eligible)
    names = e.capture.astype(str).str.lower()
    mask = names.str.contains("goldeneye|slowloris|tcpconnect").to_numpy()
    pool = valid[~mask[valid]]
    test = valid[mask[valid]]
    assert set(e.iloc[pool].label) == set(e.iloc[test].label)
    emit(
        "5G-NIDD_tool_split",
        dict(
            train_support=e.iloc[pool].label.value_counts().to_dict(),
            test_support=e.iloc[test].label.value_counts().to_dict(),
            held_captures=sorted(e.iloc[test].capture.astype(str).unique().tolist()),
            selection="capture names fixed before model evaluation; all four classes retained",
        ),
    )
    for seed in p.SEEDS:
        lab = p.sample_labels(pool, e.label.to_numpy(str), 5, seed)
        features = p.features(a, p.reference(a, lab))
        for name, x in [
            ("TCV6", features),
            ("TCV-no-content", features[:, 1:]),
            ("FlowStats16-adapted", p.flowstats(a)),
        ]:
            fitcell(a, pool, test, "5G-NIDD", "tool_holdout", name, x, 5, seed)
    # Exact representation collisions are descriptive, not proof of leakage.
    for split, pool, test in p.splits(a, "5G-NIDD"):
        X = p.features(
            a, p.reference(a, p.sample_labels(pool, e.label.to_numpy(str), 5, 42))
        )
        trainhash = pd.util.hash_pandas_object(
            pd.DataFrame(X[pool]), index=False
        ).to_numpy()
        testhash = pd.util.hash_pandas_object(
            pd.DataFrame(X[test]), index=False
        ).to_numpy()
        emit(
            "5G-NIDD_duplicates_" + split,
            dict(
                train=len(pool),
                test=len(test),
                test_exact_feature_matches=int(np.isin(testhash, trainhash).sum()),
                description="exact six-feature vector duplicates; identical summaries do not establish duplicate packets",
            ),
        )
    stems(a, "5G-NIDD")


def windows():
    rec, content = p.load_records("5G-NIDD")
    # Prefix pilot could not retain four classes; use the full records, with
    # missing-class windows explicitly reported rather than changing the task.
    sub = rec
    emit(
        "window_design",
        dict(
            records=len(sub),
            per_capture=sub.groupby("capture", observed=True).size().to_dict(),
            selection="all released packets; prefix pilot lacked four-class support and produced no reported scores",
            windows_seconds=[1, 10, 60],
            split="BS1 training / BS2 test",
            budget_percent=5,
            seeds=p.SEEDS,
            delay="first malicious packet in an original edge to first correctly malicious classified window closure; undetected edges reported separately; excludes parsing and runtime scheduling",
        ),
    )
    del rec
    gc.collect()
    for window in (1, 10, 60):
        a = p.aggregate(sub, content, "5G-NIDD", window=window)
        e = a["edges"]
        valid = np.flatnonzero(a["keep"])
        m = e.capture.astype(str).str.contains("_BS2_").to_numpy()
        pool = valid[~m[valid]]
        test = valid[m[valid]]
        y = e.label.to_numpy(str)
        classes = sorted(set(y[pool]))
        if len(classes) != 4 or set(y[test]) != set(classes):
            emit(
                f"5G-NIDD_window{window}_unavailable",
                dict(
                    kind="window_unavailable",
                    window_seconds=window,
                    train_support=e.iloc[pool].label.value_counts().to_dict(),
                    test_support=e.iloc[test].label.value_counts().to_dict(),
                    reason="five-record windows do not preserve all four classes in both partitions",
                ),
            )
            del a
            gc.collect()
            continue
        testrecord = sub.capture.astype(str).str.contains("_BS2_").to_numpy()
        raw = sub.loc[testrecord]
        rawedge = a["record_edge"][testrecord]
        # Original edge identity is retained for onset and missed-event accounting.
        origin = pd.MultiIndex.from_frame(raw[p.KEY])
        origin_code, _ = pd.factorize(origin, sort=False)
        malicious = raw.label.astype(str).to_numpy() != "Normal"
        start = (
            pd.Series(raw.ts.to_numpy()[malicious])
            .groupby(origin_code[malicious])
            .min()
        )
        for seed in p.SEEDS:
            tag = f"5G-NIDD_window{window}_s{seed}"
            if (OUT / (tag + ".json")).exists():
                continue
            lab = p.sample_labels(pool, y, 5, seed)
            ref = p.reference(a, lab)
            X = p.features(a, ref)
            clf = p.model("ExtraTrees", seed)
            clf.fit(X[lab], y[lab])
            pred = clf.predict(X[test])
            lookup = np.full(len(e), "ABSTAIN", dtype="U32")
            lookup[test] = pred
            recordpred = lookup[rawedge]
            detected = malicious & (recordpred != "Normal") & (recordpred != "ABSTAIN")
            closure = (e.iloc[rawedge].window.to_numpy() + 1) * window
            first = pd.Series(closure[detected]).groupby(origin_code[detected]).min()
            delays = (first - start.reindex(first.index)).to_numpy()
            result = dict(
                kind="window",
                window_seconds=window,
                seed=seed,
                n_train=len(pool),
                n_test=len(test),
                n_labeled=len(lab),
                raw_test_records=len(raw),
                record_coverage=float(np.mean(recordpred != "ABSTAIN")),
                record_macro_f1=float(
                    p.f1_score(
                        raw.label.astype(str),
                        recordpred,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
                attack_edges=len(start),
                detected_attack_edges=len(first),
                undetected_attack_edges=len(start) - len(first),
                closure_delay_median=float(np.median(delays)) if len(delays) else None,
                closure_delay_p95=float(np.percentile(delays, 95))
                if len(delays)
                else None,
                **p.score(y[test], pred, classes),
            )
            emit(tag, result)
            np.savez_compressed(
                OUT / (tag + ".npz"),
                labeled=lab,
                test=test,
                true=y[test],
                pred=pred,
                raw_true=raw.label.astype(str).to_numpy(),
                raw_pred=recordpred,
                delays=delays,
            )
        del a
        gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ton", "fiveg", "windows"])
    mode = parser.parse_args().mode
    start = time.perf_counter()
    globals()[mode]()
    emit(
        "completion_" + mode,
        dict(
            elapsed_seconds=time.perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
