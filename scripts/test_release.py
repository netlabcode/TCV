# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Check retained packet parsing and report utilities using temporary fixtures.

Requires requirements-experiments.txt (Scapy). No public dataset is needed.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from summarize import summarize

sys.path.insert(0, str(Path(__file__).parent / "legacy_preparation"))
import T2_0_payload_extractor as parser


def main():
    parser._import_scapy()
    packet = (
        parser.Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
        / parser.IP(src="192.0.2.1", dst="192.0.2.2")
        / parser.TCP(sport=1234, dport=80)
        / parser.Raw(b"abc")
    )
    packet.time = 1.5
    row = parser.packet_to_row(
        packet, "5G-NIDD", "synthetic", 0, 0.0, 64, "benign", "Normal", "normal"
    )
    array = np.array([row], dtype=parser.build_packet_dtype(64))
    assert array["timestamp"][0] == 1.5
    assert array["dst_port"][0] == 80 and array["l4_proto"][0] == 6
    assert array["payload"][0, :3].tolist() == [97, 98, 99]
    assert not array["payload"][0, 3:].any()
    assert array["frame_len"][0] == len(packet)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        result_dir = root / "results" / "synthetic"
        result_dir.mkdir(parents=True)
        for seed, score in enumerate([0.6, 0.7, 0.8, 0.9, 1.0]):
            result = dict(
                dataset="synthetic",
                split="fixed",
                budget=5,
                representation="TCV6",
                learner="ExtraTrees",
                seed=seed,
                clean=dict(macro_f1=score, accuracy=score),
            )
            (result_dir / f"{seed}.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
        summary = summarize(root)
        assert len(summary) == 1 and summary.n_seeds.iloc[0] == 5
        assert np.isclose(summary.macro_f1_mean.iloc[0], 0.8)
        # t(4, 0.975) * sample standard deviation / sqrt(5).
        assert np.isclose(summary.macro_f1_ci95_halfwidth.iloc[0], 0.1963243161)
        inputs = root / "inputs"
        inputs.mkdir()
        payload = b"a,b\n1,2\n"
        (inputs / "sample.csv").write_bytes(payload)
        output = root / "manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("input_manifest.py")),
                str(inputs),
                "--output",
                str(output),
            ],
            check=True,
        )
        entry = json.loads(output.read_text())["files"][0]
        assert entry["path"] == "sample.csv"
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    print("PASS: packet parser, five-seed Student-t summary and input hashing")


if __name__ == "__main__":
    main()
