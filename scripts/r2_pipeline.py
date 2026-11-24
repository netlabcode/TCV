# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Isolated second-round pipeline. All learned references use labeled training edges.

Prepared packet records are used without modifying the original shards. This is
not a PCAP parser benchmark. TON-IoT has no timestamps: never invent them.
"""

import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[k] = "1"
from pathlib import Path
import argparse, json, time, hashlib, platform, sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = Path(os.environ.get("IOT_DATA_ROOT", "data_root")).resolve()
WORK = Path(os.environ.get("IOT_REVISION_ROOT", "revision_run")).resolve()
SEEDS = [42, 123, 456, 789, 101112]
COMP = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
KEY = ["capture", "proto", "src", "dst", "port"]
CAT = {
    "normal": "Normal",
    "dos": "Flooding",
    "ddos": "Flooding",
    "scanning": "Scanning",
    "backdoor": "Exploit",
    "injection": "Exploit",
    "password": "Exploit",
    "xss": "Exploit",
    "ransomware": "Exploit",
    "mitm": "Exploit",
}


def savejson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
    )


def load_records(ds):
    """Load one dataset and cache normalized records/content under WORK.

    Content row i corresponds to record_id i. Existing caches are reused;
    callers must choose a fresh WORK directory after changing input data.
    """
    cache = WORK / "cache" / ds
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "records.pkl").exists():
        return pd.read_pickle(cache / "records.pkl"), np.load(
            cache / "content.npy", mmap_mode="r"
        )
    start = time.perf_counter()
    if ds == "TON-IoT":
        d = pd.read_csv(
            ROOT / "data/v5_raw/TON_IoT/train_test_network.csv", low_memory=False
        )

        def n(k):
            return pd.to_numeric(d[k], errors="coerce").fillna(0).to_numpy(float)

        pk = np.maximum(n("src_pkts") + n("dst_pkts"), 1)
        duration = np.maximum(n("duration"), 0)
        x = np.column_stack(
            [
                np.log1p(n("http_request_body_len")),
                np.log1p(n("http_response_body_len")),
                n("http_status_code") / 100,
                n("dns_qtype"),
                n("dns_rcode"),
                (d.ssl_established.astype(str) == "T").astype(float),
                np.log1p(n("missed_bytes")),
                np.log1p(n("src_bytes")),
                np.log1p(n("dst_bytes")),
            ]
        )
        rec = pd.DataFrame(
            {
                "capture": "toniot",
                "proto": d.proto.map({"tcp": 6, "udp": 17, "icmp": 1})
                .fillna(0)
                .astype(int),
                "src": d.src_ip.astype(str),
                "dst": d.dst_ip.astype(str),
                "port": n("dst_port").astype(int),
                "size": np.clip((n("src_ip_bytes") + n("dst_ip_bytes")) / pk, 0, 1500),
                "iat": duration / pk,
                "packet_weight": pk,
                "ts": np.nan,
                "duration": duration,
                "l7": d.service.where(d.service != "-", d.proto).astype(str),
                "label": d.type.str.lower().map(CAT),
                "attack": d.type.str.lower(),
                "sport": n("src_port").astype(int),
                "flags": 0,
                "ttl": 0,
            }
        )
        assert rec.label.notna().all()
    else:
        paths = sorted((ROOT / "features/payload_v2_4class/5G-NIDD").glob("*.npy"))
        if not paths:
            raise FileNotFoundError(
                "No prepared 5G-NIDD shards. Follow docs/DATASETS.md first."
            )
        total = sum(len(np.load(p, mmap_mode="r")) for p in paths)
        x = np.lib.format.open_memmap(
            cache / "content.npy", mode="w+", dtype=np.uint8, shape=(total, 64)
        )
        frames = []
        offset = 0
        for p in paths:
            a = np.load(p, mmap_mode="r")
            count = len(a)
            x[offset : offset + count] = a["payload"][:, :64]
            offset += count
            decode = lambda k: pd.Categorical(a[k].astype(str))
            frames.append(
                pd.DataFrame(
                    {
                        "capture": decode("pcap_id"),
                        "proto": a["l4_proto"],
                        "src": decode("src_ip"),
                        "dst": decode("dst_ip"),
                        "port": a["dst_port"],
                        "sport": a["src_port"],
                        "size": a["frame_len"].astype(float),
                        "iat": a["beta"],
                        "packet_weight": 1.0,
                        "ts": a["timestamp"],
                        "duration": 0.0,
                        "l7": decode("l7_label"),
                        "label": decode("attack_category"),
                        "attack": decode("attack_type"),
                        "flags": a["tcp_flags"],
                        "ttl": a["ip_ttl"],
                    }
                )
            )
            print("loaded", p.name, count, flush=True)
        rec = pd.concat(frames, ignore_index=True)
        x.flush()
    rec = rec.reset_index(drop=True)
    for col in ["capture", "src", "dst", "l7", "label", "attack"]:
        rec[col] = rec[col].astype("category")
    rec["record_id"] = np.arange(len(rec))
    rec.to_pickle(cache / "records.pkl")
    if ds == "TON-IoT":
        np.save(cache / "content.npy", x)
    savejson(
        cache / "loading.json",
        {
            "records": len(rec),
            "seconds": time.perf_counter() - start,
            "timestamp_available": ds == "5G-NIDD",
            "input": "prepared packet shards" if ds == "5G-NIDD" else "released CSV",
        },
    )
    return rec, np.load(cache / "content.npy", mmap_mode="r")


def aggregate(rec, content, ds, window=0, rotation=0):
    """Return raw edge sufficient statistics plus record-to-edge mapping.

    Recalculate packet IAT within each newly constructed edge, so source rotation
    and time windows actually change the timing distribution. Flow IAT remains
    the released per-flow duration/packet-count proxy.
    """
    t = time.perf_counter()
    d = rec.copy()
    key = KEY.copy()
    if rotation:
        if ds == "5G-NIDD":
            order = d.sort_values(
                ["capture", "src", "ts", "record_id"], kind="stable"
            ).index
            ordinal = (
                d.loc[order]
                .groupby(["capture", "src"], observed=True)
                .cumcount()
                .reindex(d.index)
            )
        else:
            ordinal = d.groupby(["capture", "src"], observed=True).cumcount()
        d["src"] = (
            d.src.astype(str) + "|r2rotation|" + (ordinal // rotation).astype(str)
        )
    if window:
        if ds != "5G-NIDD":
            raise ValueError("TON-IoT release has no absolute timestamps")
        d["window"] = np.floor(d.ts / window).astype(np.int64)
        key += ["window"]
    code, uniques = pd.factorize(pd.MultiIndex.from_frame(d[key]), sort=False)
    d["edge"] = code
    count = len(uniques)
    if ds == "5G-NIDD":
        ordered = d.sort_values(["edge", "ts", "record_id"], kind="stable")
        d["iat"] = (
            ordered.groupby("edge", sort=False)
            .ts.diff()
            .reindex(d.index)
            .fillna(0)
            .clip(lower=0)
        )
    gb = d.groupby("edge", sort=True, observed=True)
    e = gb.agg(
        n_records=("size", "size"),
        n_packets=("packet_weight", "sum"),
        n_bytes=("size", "sum"),
        size_mean=("size", "mean"),
        size_std=("size", "std"),
        size_min=("size", "min"),
        size_max=("size", "max"),
        duration_sum=("duration", "sum"),
        ttl_mean=("ttl", "mean"),
        flags_mean=("flags", "mean"),
        ts_min=("ts", "min"),
        ts_max=("ts", "max"),
    )
    ids = uniques.to_frame(index=False)
    ids.columns = key
    ids.index = e.index
    e = pd.concat([ids, e], axis=1)
    positive = d[d["iat"] > 0].groupby("edge", sort=True, observed=True)
    for name, agg in [
        ("iat_median", "median"),
        ("iat_mean", "mean"),
        ("iat_std", "std"),
        ("iat_min", "min"),
        ("iat_max", "max"),
    ]:
        e[name] = positive["iat"].agg(agg).reindex(e.index).fillna(0)
    # Trace convention: variances of paired size and positive timing observations.
    e["gamma_raw"] = (
        (positive["size"].var() + positive["iat"].var()).reindex(e.index).fillna(0)
    )
    e["size_std"] = e.size_std.fillna(0)
    labels = sorted(rec.label.astype(str).unique())
    labelcodes = pd.Categorical(d.label, categories=labels).codes
    counts = np.bincount(
        code * len(labels) + labelcodes, minlength=count * len(labels)
    ).reshape(count, len(labels))
    e["label"] = np.array(labels)[counts.argmax(axis=1)]
    e["purity"] = counts.max(axis=1) / e.n_records
    cats = sorted(d.l7.astype(str).unique())
    l7code = pd.Categorical(d.l7, categories=cats).codes
    l7 = sparse.coo_matrix(
        (np.ones(len(code), dtype=np.int64), (code, l7code)), shape=(count, len(cats))
    ).tocsr()
    sizes = d["size"].to_numpy()
    valid = (sizes >= 0) & (sizes <= 1500)
    bins = np.minimum((sizes[valid] / (1500 / 32)).astype(int), 31)
    hist = np.bincount(code[valid] * 32 + bins, minlength=count * 32).reshape(count, 32)
    # Preserve original histogram support; explicitly track observations outside it.
    e["size_outside_hist"] = np.bincount(code[~valid], minlength=count)
    means = np.empty((count, content.shape[1]), float)
    idx = d.record_id.to_numpy()
    for j in range(content.shape[1]):
        means[:, j] = (
            np.bincount(code, weights=content[idx, j], minlength=count) / e.n_records
        )
    keep = e.n_records.to_numpy() >= 5
    # Minimum support now consistently means five traffic records. Historical TON
    # code instead used summed packet weights; record this scientific correction.
    print(
        "aggregate",
        ds,
        "window",
        window,
        "rotation",
        rotation,
        "records",
        len(d),
        "edges",
        count,
        "retained",
        int(keep.sum()),
        "seconds",
        round(time.perf_counter() - t, 2),
        flush=True,
    )
    return {
        "edges": e,
        "content": means,
        "hist": hist,
        "l7": l7,
        "l7_names": cats,
        "keep": keep,
        "record_edge": code,
        "record_id": idx,
        "iat": d["iat"].to_numpy(),
        "record_label": d.label.astype(str).to_numpy(),
        "seconds": time.perf_counter() - t,
    }


def reference(a, train):
    """Fit benign distributions from annotated training edge indices only.

    `train` must contain the annotation subset, not all eligible training
    edges. The corpus-reference sensitivity experiment is an explicit exception.
    """
    e = a["edges"]
    benign = np.asarray(train)[e.iloc[train].label.to_numpy() == "Normal"]
    if not len(benign):
        raise ValueError("No labeled benign training edges")
    weights = e.iloc[benign].n_records.to_numpy()
    mask = np.isin(a["record_edge"], benign)
    iat = a["iat"][mask]
    iat = iat[iat > 0]
    hist = a["hist"][benign].sum(axis=0).astype(float)
    hist /= max(hist.sum(), 1)
    l7 = np.asarray(a["l7"][benign].sum(axis=0)).ravel().astype(float)
    l7 /= max(l7.sum(), 1)
    return {
        "content": np.average(a["content"][benign], axis=0, weights=weights),
        "median": float(np.median(iat)) if len(iat) else 0.0,
        "iqr": max(float(np.subtract(*np.percentile(iat, [75, 25]))), 1e-6)
        if len(iat)
        else 1e-6,
        "hist": hist,
        "l7": dict(zip(a["l7_names"], l7)),
        "benign_edges": benign.tolist(),
        "benign_records": int(mask.sum()),
    }


def features(a, ref):
    """Return [alpha, beta, gamma, delta, epsilon, zeta] for every edge.

    Rows include ineligible edges; index with a['keep'] before classification.
    Reuse the training reference unchanged when transforming held-out records.
    """
    e = a["edges"]
    hist = a["hist"] / np.maximum(a["hist"].sum(axis=1, keepdims=True), 1)
    counts = sparse.csr_matrix(a["l7"])
    row = np.repeat(np.arange(len(e)), np.diff(counts.indptr))
    p = counts.data / np.maximum(np.asarray(counts.sum(axis=1)).ravel()[row], 1)
    q_all = np.array([ref["l7"].get(k, 0) for k in a["l7_names"]])
    q = q_all[counts.indices]
    # Exact sparse JS: baseline mass outside a row's support contributes q*log(2).
    terms = p * np.log(2 * p / (p + q))
    valid = q > 0
    terms[valid] += q[valid] * np.log(q[valid] / (p[valid] + q[valid]))
    js = np.clip(
        0.5 * (np.log(2) + np.bincount(row, weights=terms, minlength=len(e))),
        0,
        np.log(2),
    )
    beta = np.abs(e.iat_median.to_numpy() - ref["median"]) / ref["iqr"]
    beta[e.iat_median.to_numpy() == 0] = 0
    return np.column_stack(
        [
            np.abs(a["content"] - ref["content"]).max(axis=1),
            beta,
            e.gamma_raw,
            np.abs(np.cumsum(hist, axis=1) - np.cumsum(ref["hist"])).mean(axis=1),
            js,
            np.log1p(e.size_mean),
        ]
    )


def flowstats(a):
    """Explicitly an adapted 16-statistic baseline, NOT CICFlowMeter."""
    e = a["edges"]
    duration = (e.ts_max - e.ts_min).fillna(e.duration_sum).clip(lower=0)
    return np.column_stack(
        [
            e.n_records,
            e.n_packets,
            e.n_bytes,
            e.size_mean,
            e.size_std,
            e.size_min,
            e.size_max,
            e.iat_mean,
            e.iat_std,
            e.iat_min,
            e.iat_max,
            duration,
            e.n_records / np.maximum(duration, 1e-6),
            e.n_bytes / np.maximum(duration, 1e-6),
            e.flags_mean,
            e.ttl_mean,
        ]
    )


def identity(a):
    e = a["edges"]
    return np.column_stack(
        [
            pd.util.hash_array(e.src.astype(str).to_numpy()).astype(float) / 2**64,
            pd.util.hash_array(e.dst.astype(str).to_numpy()).astype(float) / 2**64,
            e.proto,
            e.port,
        ]
    )


def splits(a, ds):
    """Yield fixed random and dataset-specific group splits of eligible edges."""
    e = a["edges"]
    valid = np.flatnonzero(a["keep"])
    y = e.label.to_numpy()
    tr, te = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2026).split(
            valid, y[valid]
        )
    )
    yield "random_fixed20", valid[tr], valid[te]
    if ds == "5G-NIDD":
        for bs in ["BS1", "BS2"]:
            mask = e.capture.astype(str).str.contains("_" + bs + "_").to_numpy()
            yield "holdout_" + bs, valid[~mask[valid]], valid[mask[valid]]
    else:
        # Metadata-only allocation: put one major flooding/scanning source on each
        # side, distribute benign sources deterministically, and include Exploit.
        groups = sorted(e.src.astype(str).unique())
        held = set(groups[::2])
        held.update(["192.168.1.30", "192.168.1.33", "192.168.1.34"])
        held.difference_update(["192.168.1.31", "192.168.1.32", "192.168.1.39"])
        mask = e.src.astype(str).isin(held).to_numpy()
        for name, m in [("A", mask), ("B", ~mask)]:
            yield "source_holdout_" + name, valid[~m[valid]], valid[m[valid]]


def sample_labels(pool, y, budget, seed):
    """Sample an exact percentage budget while retaining each training class."""
    n = int(np.floor(len(pool) * budget / 100))
    if n < len(np.unique(y[pool])):
        raise ValueError("budget cannot cover classes")
    idx, _ = next(
        StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed).split(
            pool, y[pool]
        )
    )
    selected = pool[idx].copy()
    # At tiny budgets ordinary proportional stratification can omit a rare class.
    # Preserve the exact budget while guaranteeing one annotation per class.
    rng = np.random.default_rng(seed)
    for missing in sorted(set(y[pool]) - set(y[selected])):
        vals, counts = np.unique(y[selected], return_counts=True)
        donor = vals[np.argmax(counts)]
        assert counts.max() > 1
        replace = rng.choice(np.flatnonzero(y[selected] == donor))
        selected[replace] = rng.choice(pool[y[pool] == missing])
    assert len(np.unique(selected)) == n and set(y[selected]) == set(y[pool])
    return selected


def model(name, seed):
    if name == "GradBoost":
        return GradientBoostingClassifier(random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=120, n_jobs=1, random_state=seed)
    return RandomForestClassifier(
        n_estimators=80, max_depth=10, n_jobs=1, random_state=seed
    )


def score(y, yp, classes):
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=classes, zero_division=0)
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float(accuracy_score(y, yp)),
        "per_class": {
            c: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y, yp, labels=classes).tolist(),
    }


def run(ds, pilot=False):
    """Execute the primary experiment grid and write per-cell metrics/predictions."""
    rec, content = load_records(ds)
    out = WORK / ("pilot_results" if pilot else "results") / ds
    out.mkdir(parents=True, exist_ok=True)
    cache = WORK / "cache" / ds / "aggregate_v2_sparse.pkl"
    if cache.exists():
        a = pd.read_pickle(cache)
    else:
        a = aggregate(rec, content, ds)
        pd.to_pickle(a, cache)
    e = a["edges"]
    y = e.label.to_numpy(dtype=str)
    classes = sorted(set(y[a["keep"]]))
    design = list(splits(a, ds))
    savejson(
        out / "design.json",
        {
            "classes": classes,
            "support_definition": "at least 5 records",
            "budget_denominator": "eligible training-pool edges; fixed test set",
            "splits": [
                {
                    "name": name,
                    "train_support": e.iloc[tr].label.value_counts().to_dict(),
                    "test_support": e.iloc[te].label.value_counts().to_dict(),
                    "training_edges": tr.tolist(),
                    "test_edges": te.tolist(),
                }
                for name, tr, te in design
            ],
        },
    )
    for split, pool, test in design:
        assert set(y[pool]) == set(classes) and set(y[test]) == set(classes), (
            split,
            "missing class",
        )
        # Filter raw test records by the clean edge membership once. Aggregation
        # happens AFTER test-only source rotation, never on mixed train/test data.
        if pilot and split != "random_fixed20":
            continue
        variants = {0: a}
        rotation_levels = [1000, 100, 10] if not pilot else [100]
        rawmask = np.isin(a["record_edge"], test)
        rawtest = rec.loc[rawmask]
        for rotation in rotation_levels:
            rp = WORK / "cache" / ds / f"{split}_rotation{rotation}_v2_sparse.pkl"
            if rp.exists():
                variants[rotation] = pd.read_pickle(rp)
            else:
                variants[rotation] = aggregate(rawtest, content, ds, rotation=rotation)
                pd.to_pickle(variants[rotation], rp)
        for budget in [5] if pilot else [1, 3, 5]:
            for seed in [42] if pilot else SEEDS:
                labeled = sample_labels(pool, y, budget, seed)
                ref = reference(a, labeled)
                x = features(a, ref)
                configs = {
                    "TCV6": x,
                    "TCV3": x[:, :3],
                    "Identity": identity(a),
                    "FlowStats16-adapted": flowstats(a),
                }
                if not pilot and budget == 5:
                    for j, c in enumerate(COMP):
                        configs["only_" + c] = x[:, [j]]
                        configs["drop_" + c] = np.delete(x, j, axis=1)
                for name, X in configs.items():
                    for learner in (
                        ["GradBoost", "ExtraTrees"]
                        if name in ["TCV6", "TCV3", "FlowStats16-adapted"]
                        else ["RandomForest"]
                        if name == "Identity"
                        else ["GradBoost"]
                    ):
                        tag = f"{split}_b{budget}_s{seed}_{name}_{learner}"
                        path = out / (tag + ".json")
                        if path.exists():
                            previous = np.load(out / (tag + "_predictions.npz"))
                            same = np.array_equal(previous["labeled_edge"], labeled)
                            previous.close()
                            if same:
                                continue
                            # Resume must not silently reuse an earlier label-sampling
                            # protocol. Preserve superseded cells outside the result tree.
                            archive = WORK / "superseded" / ds
                            archive.mkdir(parents=True, exist_ok=True)
                            suffix = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
                            path.rename(archive / (tag + "_" + suffix + ".json"))
                            (out / (tag + "_predictions.npz")).rename(
                                archive / (tag + "_" + suffix + "_predictions.npz")
                            )
                        clf = model(learner, seed)
                        t = time.perf_counter()
                        clf.fit(X[labeled], y[labeled])
                        fit = time.perf_counter() - t
                        t = time.perf_counter()
                        pred = clf.predict(X[test])
                        infer = time.perf_counter() - t
                        result = {
                            "dataset": ds,
                            "split": split,
                            "budget": budget,
                            "seed": seed,
                            "representation": name,
                            "learner": learner,
                            "n_labeled": len(labeled),
                            "labeled_support": e.iloc[labeled]
                            .label.value_counts()
                            .to_dict(),
                            "fit_seconds": fit,
                            "inference_seconds": infer,
                            "classes": classes,
                            "reference_benign_edges": ref["benign_edges"],
                            "reference_benign_records": ref["benign_records"],
                            "clean": score(y[test], pred, classes),
                            "rotation": {},
                        }
                        clean_lookup = np.full(len(e), "ABSTAIN", dtype="U32")
                        clean_lookup[test] = pred
                        raw_clean = clean_lookup[a["record_edge"][rawmask]]
                        raw_true = a["record_label"][rawmask]
                        result["clean"].update(
                            record_coverage=1.0,
                            record_macro_f1=float(
                                f1_score(
                                    raw_true,
                                    raw_clean,
                                    labels=classes,
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                            record_accuracy=float(np.mean(raw_true == raw_clean)),
                            raw_records=len(raw_true),
                        )
                        predarrays = {
                            "test_edge": test,
                            "true": y[test],
                            "pred": pred,
                            "labeled_edge": labeled,
                        }
                        for rotation, v in variants.items():
                            if rotation == 0 or name not in ["TCV6", "Identity"]:
                                continue
                            vr = np.flatnonzero(v["keep"])
                            ve = v["edges"]
                            Xr = features(v, ref) if name == "TCV6" else identity(v)
                            yp = (
                                clf.predict(Xr[vr])
                                if len(vr)
                                else np.array([], dtype=str)
                            )
                            rr = (
                                score(ve.iloc[vr].label.to_numpy(), yp, classes)
                                if len(vr)
                                else {"macro_f1": None}
                            )
                            lookup = np.full(len(ve), "ABSTAIN", dtype="U32")
                            lookup[vr] = yp
                            rawpred = lookup[v["record_edge"]]
                            rr.update(
                                retained_edges=len(vr),
                                all_edges=len(ve),
                                record_coverage=float(np.mean(rawpred != "ABSTAIN")),
                                record_macro_f1=float(
                                    f1_score(
                                        v["record_label"],
                                        rawpred,
                                        labels=classes,
                                        average="macro",
                                        zero_division=0,
                                    )
                                ),
                                record_accuracy=float(
                                    np.mean(v["record_label"] == rawpred)
                                ),
                            )
                            result["rotation"][str(rotation)] = rr
                            predarrays["rotation" + str(rotation) + "_edge"] = vr
                            predarrays["rotation" + str(rotation) + "_pred"] = yp
                        np.savez_compressed(
                            out / (tag + "_predictions.npz"), **predarrays
                        )
                        savejson(path, result)
                        print(
                            "DONE", tag, "f1", result["clean"]["macro_f1"], flush=True
                        )
        if pilot:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["5G-NIDD", "TON-IoT"], required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    savejson(
        WORK / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "threads": 1,
        },
    )
    if args.prepare:
        r, c = load_records(args.dataset)
        a = aggregate(r, c, args.dataset)
        pd.to_pickle(a, WORK / "cache" / args.dataset / "aggregate_v2_sparse.pkl")
    else:
        run(args.dataset, args.pilot)
