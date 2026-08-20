"""Write portable SHA-256 checksums for the staged release snapshot."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "manifests" / "SHA256SUMS.txt"


def main() -> None:
    listed = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    rows = []
    for relative in sorted(set(listed)):
        path = ROOT / relative
        if not path.is_file() or path.resolve() == OUTPUT.resolve():
            continue
        canonical_bytes = subprocess.check_output(
            ["git", "show", f":{relative}"], cwd=ROOT
        )
        digest = hashlib.sha256(canonical_bytes).hexdigest()
        normalized = relative.replace("\\", "/")
        rows.append(f"{digest}  {normalized}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(rows)} entries")


if __name__ == "__main__":
    main()
