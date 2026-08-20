"""Download or validate the exact public data snapshots used in NeuroTx-MOOD."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "data_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checked(destination: Path, url: str, expected: str, force: bool) -> None:
    if destination.exists() and not force and sha256_file(destination) == expected:
        print(f"verified {destination.relative_to(ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroTx-MOOD/1.0 data preparation"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    observed = sha256_file(partial)
    if observed != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {destination.name}: {observed}")
    partial.replace(destination)
    print(f"downloaded and verified {destination.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload public inputs")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    for relative, metadata in manifest.items():
        destination = ROOT / relative
        expected = metadata["sha256"]
        if metadata.get("redistributed", False):
            if not destination.exists():
                raise FileNotFoundError(f"release file missing: {relative}")
            observed = sha256_file(destination)
            if observed != expected:
                raise ValueError(f"checksum mismatch for {relative}: {observed}")
            print(f"verified {relative}")
        else:
            download_checked(destination, metadata["source_locator"], expected, args.force)


if __name__ == "__main__":
    main()
