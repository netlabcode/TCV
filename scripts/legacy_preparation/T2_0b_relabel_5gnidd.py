# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""T2.0b: Re-extract 5G-NIDD packets WITH per-packet benign/attack labels.

The original payload extractor (T2_0_payload_extractor.py :: enumerate_5gnidd)
labelled every packet in an attack-named PCAP as "attack" (filename-based).
That discarded the benign background traffic that 5G-NIDD captures contain,
so the downstream task collapsed to 3-class attack categorisation.

This script restores the benign class by joining each packet to the
5G-NIDD Argus per-flow CSVs, which carry a per-5-tuple ``Label`` column
(Benign / Malicious). We use the GTP-removed (inner-IP) PCAPs so that the
packet 5-tuples match the Argus inner-flow 5-tuples.

Output: sharded .npy per PCAP into features/payload_v2_4class/5G-NIDD/,
using the SAME dtype as T2_0 so the existing T2_1 aggregator can consume it.

Usage:
  python3 T2_0b_relabel_5gnidd.py validate   # one PCAP, prints benign recovery
  python3 T2_0b_relabel_5gnidd.py run         # all nogtp PCAPs -> shards
"""

from __future__ import annotations
import os, sys, time, re, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(os.environ["IOT_DATA_ROOT"])
NIDD = Path(os.environ["FIVEG_RAW_ROOT"])
OUT_DIR = ROOT / "features/payload_v2_4class/5G-NIDD"
LOG = ROOT / "features/logs/T2_0b_relabel.log"
N_PAYLOAD = 256

sys.path.insert(0, str(Path(__file__).resolve().parent))
import T2_0_payload_extractor as T2  # reuse dtype + packet_to_row + scapy import

ATTACK_CATEGORY = {
    "goldeneye": "Flooding",
    "synflood": "Flooding",
    "udpflood": "Flooding",
    "icmpflood": "Flooding",
    "ssh": "Flooding",
    "synscan": "Scanning",
    "tcpconnect": "Scanning",
    "udpscan": "Scanning",
    "slowloris": "Slow-rate",
    "torshammer": "Slow-rate",
}

PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1, "sctp": 132, "igmp": 2}
SERVICE_PORT = {  # common Argus service-name encodings seen in 5G-NIDD
    "https": 443,
    "http": 80,
    "domain": 53,
    "ssh": 22,
    "ftp": 21,
    "telnet": 23,
    "smtp": 25,
    "ntp": 123,
    "snmp": 161,
}


def log(msg):
    line = f"[{datetime.utcnow().isoformat()}Z] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def norm_port(v):
    """Argus encodes ports as decimal, 0x-hex, or service name."""
    if v is None:
        return -1
    s = str(v).strip().lower()
    if s == "" or s == "nan":
        return -1
    if s in SERVICE_PORT:
        return SERVICE_PORT[s]
    try:
        if s.startswith("0x"):
            return int(s, 16)
        return int(float(s))
    except ValueError:
        return -1


def build_label_map(csv_path: Path):
    """Return dict keyed by frozenset({ipA,ipB}) + proto -> 'attack'/'normal'.
    Bidirectional (flows are symmetric for labelling). If a pair carries both
    Benign and Malicious flows, the malicious label wins (conservative:
    attack traffic defines the edge), and we count such collisions.
    """
    a = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=[
            "SrcAddr",
            "DstAddr",
            "Proto",
            "Sport",
            "Dport",
            "Label",
            "Attack Tool",
        ],
    )
    a = a.dropna(subset=["SrcAddr", "DstAddr", "Proto"])
    lab_map = {}
    collisions = 0
    for r in a.itertuples(index=False):
        proto = PROTO_NUM.get(str(r.Proto).strip().lower(), -1)
        key = (proto, frozenset((str(r.SrcAddr), str(r.DstAddr))))
        is_atk = str(r.Label).strip().lower() == "malicious"
        if key in lab_map:
            if lab_map[key][0] != is_atk:
                collisions += 1
                if is_atk:  # malicious wins
                    lab_map[key] = (True, str(r._6))
        else:
            lab_map[key] = (is_atk, str(r._6))
    return lab_map, collisions, len(a)


def csv_for(attack_type: str, bs: str):
    """Map pcap attack+BS to the Argus per-attack CSV path."""
    # csv names: Goldeneye1.csv, ICMPFlood1.csv, SYNScan1.csv, ... (BS1->1, BS2->2)
    n = "1" if bs == "BS1" else "2"
    d = NIDD / f"csv/{bs}_each_attack"
    if not d.is_dir():
        return None
    # case-insensitive match on stem without the trailing BS digit
    want = attack_type.lower()
    alias = {
        "icmpflood": "icmpflood",
        "synflood": "synflood",
        "synscan": "synscan",
        "udpflood": "udpflood",
        "udpscan": "udpscan",
        "tcpconnect": "tcpconnect",
        "goldeneye": "goldeneye",
        "slowloris": "slowloris",
        "torshammer": "torshammer",
        "ssh": "ssh",
    }
    want = alias.get(want, want)
    for f in d.glob("*.csv"):
        stem = re.sub(r"\d+$", "", f.stem).lower()
        if stem == want:
            return f
    return None


def enumerate_nogtp():
    """Inner-IP (GTP-removed) 5G-NIDD PCAPs with their Argus CSV."""
    out = []
    for sub in ("pcap/BS1_GTP_removed", "pcap/BS2_GTP_removed"):
        d = NIDD / sub
        if not d.is_dir():
            continue
        bs = "BS1" if "BS1" in sub else "BS2"
        for pcap in sorted(d.glob("*.pcap*")):
            m = re.match(r"^([A-Za-z]+)_BS[12]_nogtp$", pcap.stem)
            if not m:
                continue
            atk = m.group(1).lower()
            csvp = csv_for(atk, bs)
            out.append(
                {
                    "pcap_path": str(pcap),
                    "pcap_id": f"5gnidd_{pcap.stem}",
                    "attack_type": atk,
                    "bs": bs,
                    "csv": str(csvp) if csvp else None,
                }
            )
    return out


def extract_one(rec, dtype):
    """Stream a PCAP, label each packet via the Argus map, return struct array."""
    T2._import_scapy()
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP, SCTP, Ether  # noqa

    lab_map = collisions = None
    if rec["csv"]:
        lab_map, collisions, n_flows = build_label_map(Path(rec["csv"]))
    base_cat = ATTACK_CATEGORY.get(rec["attack_type"], "Flooding")
    last_ts = {}
    rows = []
    n_benign = n_attack = n_unmatched = 0
    with PcapReader(rec["pcap_path"]) as reader:
        for i, pkt in enumerate(reader):
            try:
                if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
                    continue
                ts = float(pkt.time)
                if pkt.haslayer(IP):
                    s, d = pkt[IP].src, pkt[IP].dst
                else:
                    s, d = pkt[IPv6].src, pkt[IPv6].dst
                if pkt.haslayer(TCP):
                    proto, dp = 6, int(pkt[TCP].dport)
                elif pkt.haslayer(UDP):
                    proto, dp = 17, int(pkt[UDP].dport)
                elif pkt.haslayer(ICMP):
                    proto, dp = 1, -1
                elif pkt.haslayer(SCTP):
                    proto, dp = 132, int(pkt[SCTP].dport)
                else:
                    proto, dp = (
                        int(pkt[IP].proto if pkt.haslayer(IP) else pkt[IPv6].nh),
                        -1,
                    )
                # label via Argus bidirectional (proto, {src,dst})
                is_atk = True
                if lab_map is not None:
                    hit = lab_map.get((proto, frozenset((s, d))))
                    if hit is None:
                        n_unmatched += 1
                        is_atk = (
                            False  # unmatched inner flows treated as benign background
                        )
                    else:
                        is_atk = hit[0]
                if is_atk:
                    bin_lab, atk_t, cat = "attack", rec["attack_type"], base_cat
                    n_attack += 1
                else:
                    bin_lab, atk_t, cat = "normal", "benign", "Normal"
                    n_benign += 1
                key = (proto, s, d, dp)
                lt = last_ts.get(key)
                beta = (ts - lt) if lt is not None else -1.0
                last_ts[key] = ts
                rows.append(
                    T2.packet_to_row(
                        pkt,
                        "5G-NIDD",
                        rec["pcap_id"],
                        i,
                        beta,
                        N_PAYLOAD,
                        atk_t,
                        cat,
                        bin_lab,
                    )
                )
            except Exception:
                continue
    arr = np.array(rows, dtype=dtype) if rows else np.zeros(0, dtype=dtype)
    return arr, dict(
        n_benign=n_benign,
        n_attack=n_attack,
        n_unmatched=n_unmatched,
        collisions=collisions,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["validate", "run"], nargs="?", default="validate"
    )
    mode = parser.parse_args().mode
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dtype = T2.build_packet_dtype(N_PAYLOAD)
    recs = enumerate_nogtp()
    if not recs:
        raise FileNotFoundError(
            "No GTP-removed PCAPNG captures found; see docs/DATASETS.md"
        )
    if any(not rec["csv"] for rec in recs):
        raise FileNotFoundError(
            "A capture is missing its matching Argus CSV; see docs/DATASETS.md"
        )
    log(f"mode={mode}: {len(recs)} nogtp PCAPs found")
    for r in recs:
        log(f"  {r['pcap_id']}  csv={'yes' if r['csv'] else 'MISSING'}")

    if mode == "validate":
        rec = recs[0]
        log(f"VALIDATE on {rec['pcap_id']} (csv={rec['csv']})")
        t0 = time.time()
        arr, stats = extract_one(rec, dtype)
        log(
            f"  packets={len(arr)} benign={stats['n_benign']} "
            f"attack={stats['n_attack']} unmatched={stats['n_unmatched']} "
            f"argus_collisions={stats['collisions']}  ({time.time() - t0:.0f}s)"
        )
        return 0

    # run: all PCAPs
    t0 = time.time()
    done = 0
    for rec in recs:
        outp = OUT_DIR / f"{rec['pcap_id']}.npy"
        if outp.exists() and outp.stat().st_size > 0:
            log(f"  skip existing {rec['pcap_id']}")
            done += 1
            continue
        try:
            arr, stats = extract_one(rec, dtype)
            if arr.size == 0:
                log(f"  EMPTY {rec['pcap_id']}")
                continue
            np.save(outp, arr, allow_pickle=False)
            done += 1
            log(
                f"  [{done}/{len(recs)}] {rec['pcap_id']}: pkts={len(arr)} "
                f"benign={stats['n_benign']} attack={stats['n_attack']} "
                f"unmatched={stats['n_unmatched']}  ({time.time() - t0:.0f}s)"
            )
            gc.collect()
        except Exception as e:
            import traceback

            log(f"  FAIL {rec['pcap_id']}: {e}\n{traceback.format_exc()[:300]}")
    log(f"=== T2_0b DONE: {done}/{len(recs)} shards in {time.time() - t0:.0f}s ===")
    return 0 if done == len(recs) else 1


if __name__ == "__main__":
    sys.exit(main())
