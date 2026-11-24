# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""T2.0c: Fast post-hoc relabel of the re-extracted 5G-NIDD shards.

Fixes the label rule used in T2_0b: the 5G-NIDD Argus per-attack CSVs encode
the malicious class inconsistently -- some files use the literal string
``Malicious`` while others use the attack NAME (e.g. ``UDPFlood``,
``HTTPFlood``). The correct rule is therefore: a flow is attack iff its
``Label`` is anything other than ``Benign``.

We do NOT re-parse the PCAPs. The shards already store each packet's
(l4_proto, src_ip, dst_ip), so we rebuild the Argus flow-label map with the
corrected rule and re-assign binary_label / attack_type / attack_category
in place. Pure numpy/pandas, seconds per shard.
"""

from __future__ import annotations
import os, sys, re, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ["IOT_DATA_ROOT"])
SHARD_DIR = ROOT / "features/payload_v2_4class/5G-NIDD"
NIDD = Path(os.environ["FIVEG_RAW_ROOT"])

PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1, "sctp": 132, "igmp": 2}
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


def csv_for(attack_type, bs):
    d = NIDD / f"csv/{bs}_each_attack"
    if not d.is_dir():
        return None
    for f in d.glob("*.csv"):
        if re.sub(r"\d+$", "", f.stem).lower() == attack_type.lower():
            return f
    return None


def build_map(csv_path):
    a = pd.read_csv(
        csv_path, low_memory=False, usecols=["SrcAddr", "DstAddr", "Proto", "Label"]
    )
    a = a.dropna(subset=["SrcAddr", "DstAddr", "Proto"])
    m = {}
    for r in a.itertuples(index=False):
        proto = PROTO_NUM.get(str(r.Proto).strip().lower(), -1)
        key = (proto, frozenset((str(r.SrcAddr), str(r.DstAddr))))
        is_atk = str(r.Label).strip().lower() != "benign"
        # attack wins on any collision
        m[key] = m.get(key, False) or is_atk
    return m


def main():
    shards = sorted(SHARD_DIR.glob("5gnidd_*_nogtp.npy"))
    if not shards:
        raise FileNotFoundError(
            "No generated shards found; run T2_0b_relabel_5gnidd.py run first"
        )
    for shard in shards:
        match = re.match(r"5gnidd_([A-Za-z]+)_(BS[12])_nogtp", shard.stem)
        if match is None or csv_for(match.group(1).lower(), match.group(2)) is None:
            raise FileNotFoundError(
                f"Missing matching Argus CSV or invalid shard name: {shard.name}"
            )
    print(f"{len(shards)} shards")
    tot_b = tot_a = 0
    t0 = time.time()
    for sp in shards:
        m_re = re.match(r"5gnidd_([A-Za-z]+)_(BS[12])_nogtp", sp.stem)
        attack_type = m_re.group(1).lower()
        bs = m_re.group(2)
        csvp = csv_for(attack_type, bs)
        cat = ATTACK_CATEGORY.get(attack_type, "Flooding")
        arr = np.load(sp)  # writable copy
        lab_map = build_map(csvp) if csvp else {}
        proto = arr["l4_proto"]
        sip = arr["src_ip"]
        dip = arr["dst_ip"]
        bl = arr["binary_label"]
        at = arr["attack_type"]
        ac = arr["attack_category"]
        nt = arr["normal_type"]
        nb = na = 0
        # vectorise key lookup per unique (proto, src, dst)
        keys = list(
            zip(
                proto.tolist(),
                [s.decode("latin-1") for s in sip],
                [d.decode("latin-1") for d in dip],
            )
        )
        cache = {}
        for i, (p, s, d) in enumerate(keys):
            k = (p, frozenset((s, d)))
            isa = cache.get(k)
            if isa is None:
                isa = lab_map.get(k, False)  # unmatched -> benign
                cache[k] = isa
            if isa:
                bl[i] = b"attack"
                at[i] = attack_type.encode()[:48]
                ac[i] = cat.encode()[:16]
                nt[i] = b""
                na += 1
            else:
                bl[i] = b"normal"
                at[i] = b"benign"
                ac[i] = b"Normal"
                nt[i] = b"other"
                nb += 1
        np.save(sp, arr, allow_pickle=False)
        tot_b += nb
        tot_a += na
        print(f"  {sp.stem}: benign={nb} attack={na}  ({time.time() - t0:.0f}s)")
    print(
        f"TOTAL benign={tot_b} attack={tot_a} frac_benign={tot_b / (tot_a + tot_b):.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
