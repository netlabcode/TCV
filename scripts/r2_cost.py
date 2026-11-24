# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Raw-PCAP to prediction batch benchmark; excludes model fitting, includes parsing."""

import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[k] = "1"
import sys, time, json, resource, platform, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import r2_pipeline as p

sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy_preparation"))
import T2_0_payload_extractor as extractor

extractor._import_scapy()
OUT = p.WORK / "addenda"
OUT.mkdir(parents=True, exist_ok=True)
a = pd.read_pickle(p.WORK / "cache/5G-NIDD/aggregate_v2_sparse.pkl")
e = a["edges"]
_, pool, test = next(x for x in p.splits(a, "5G-NIDD") if x[0] == "holdout_BS2")
lab = p.sample_labels(pool, e.label.to_numpy(str), 5, 42)
ref = p.reference(a, lab)
X = p.features(a, ref)
clf = p.model("ExtraTrees", 42).fit(X[lab], e.iloc[lab].label)
capture = (
    Path(os.environ["FIVEG_RAW_ROOT"])
    / "pcap/BS2_GTP_removed/Goldeneye_BS2_nogtp.pcapng"
)
rows = []
for n in (2000, 5000, 10000):
    for rep in range(3):
        tag = f"cost_{n}_{rep}"
        target = OUT / (tag + ".json")
        if target.exists():
            continue
        t = time.perf_counter()
        arr = []
        with extractor.PcapReader(str(capture)) as reader:
            for i, pkt in enumerate(reader):
                arr.append(
                    extractor.packet_to_row(
                        pkt,
                        "5G-NIDD",
                        "benchmark",
                        i,
                        0.0,
                        64,
                        "unknown",
                        "Normal",
                        "normal",
                    )
                )
                if len(arr) >= n:
                    break
        raw = np.array(arr, dtype=extractor.build_packet_dtype(64))
        content = raw["payload"].copy()
        d = pd.DataFrame(
            dict(
                capture="benchmark",
                proto=raw["l4_proto"],
                src=raw["src_ip"].astype(str),
                dst=raw["dst_ip"].astype(str),
                port=raw["dst_port"],
                sport=raw["src_port"],
                size=raw["frame_len"].astype(float),
                iat=0.0,
                packet_weight=1.0,
                ts=raw["timestamp"],
                duration=0.0,
                l7=raw["l7_label"].astype(str),
                label="Normal",
                attack="unknown",
                flags=raw["tcp_flags"],
                ttl=raw["ip_ttl"],
                record_id=np.arange(len(raw)),
            )
        )
        tparse = time.perf_counter()
        agg = p.aggregate(d, content, "5G-NIDD")
        tagg = time.perf_counter()
        fv = p.features(agg, ref)
        tf = time.perf_counter()
        keep = np.flatnonzero(agg["keep"])
        pred = clf.predict(fv[keep]) if len(keep) else []
        end = time.perf_counter()
        result = dict(
            kind="cost",
            requested_records=n,
            records=len(raw),
            repetition=rep,
            edges=len(agg["edges"]),
            retained_edges=len(keep),
            parsing_seconds=tparse - t,
            aggregation_seconds=tagg - tparse,
            feature_seconds=tf - tagg,
            prediction_seconds=end - tf,
            total_seconds=end - t,
            records_per_second=len(raw) / (end - t),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope="warm-process offline batch: PCAP read/Scapy decode/record assembly + aggregation + all TCV features + inference; model fit/imports excluded; no arrival waiting or live capture overhead",
            capture=capture.name,
            python=platform.python_version(),
            platform=platform.platform(),
            cpu=Path("/proc/cpuinfo")
            .read_text()
            .split("model name")[1]
            .split("\n")[0]
            .strip(": \t"),
            threads=1,
        )
        p.savejson(target, result)
        print(tag, result["total_seconds"], flush=True)
p.savejson(
    OUT / "completion_cost.json",
    dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repetitions=3,
        sizes=[2000, 5000, 10000],
    ),
)
