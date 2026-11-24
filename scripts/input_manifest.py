# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Hash dataset inputs with bounded memory; store relative paths, not user paths."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".npy",
            ".pcap",
            ".pcapng",
        }:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            dict(
                path=path.relative_to(root).as_posix(),
                bytes=path.stat().st_size,
                sha256=digest.hexdigest(),
            )
        )
    if not files:
        parser.error("No CSV, NPY, PCAP or PCAPNG inputs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"algorithm": "SHA-256", "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(files)} input hashes to {args.output}")


if __name__ == "__main__":
    main()
