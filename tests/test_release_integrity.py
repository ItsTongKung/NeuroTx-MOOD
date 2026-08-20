from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_json_files_parse() -> None:
    for path in ROOT.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_available_data_match_manifest() -> None:
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))["files"]
    for relative, metadata in manifest.items():
        path = ROOT / relative
        if path.exists():
            assert sha256_file(path) == metadata["sha256"]


def test_corrected_b3db_metadata_reference() -> None:
    payload = json.loads((ROOT / "b3db_metadata_probe.json").read_text(encoding="utf-8"))
    assert payload["n_rows"] == 6167
    assert payload["n_contexts"] == 11
    assert payload["n_contexts_rich"] == 15
    assert round(payload["global_cp"]["coverage_90"]["mean"], 4) == 0.7751
    assert round(payload["metadata_context_rich_cp"]["coverage_90"]["mean"], 4) == 0.8157


def test_paired_fixed_score_effects_are_retained() -> None:
    metadata = json.loads((ROOT / "b3db_metadata_probe.json").read_text(encoding="utf-8"))
    rich = metadata["paired_effects_vs_global_cp"]["metadata_context_rich_cp"]
    assert rich["coverage_90"]["n_pairs"] == 10
    assert round(rich["coverage_90"]["mean_difference"], 4) == 0.0406
    assert rich["coverage_90"]["ci95_low"] > 0.0

    external = json.loads((ROOT / "external_conformal_diagnostics.json").read_text(encoding="utf-8"))
    proxy = external["fixed_predictor_external_probes"]["paired_effects_vs_global_full_rf"]["proxy_context"]
    assert proxy["coverage"]["n_pairs"] == 10
    assert round(proxy["coverage"]["mean_difference"], 4) == 0.0304
    assert proxy["coverage"]["ci95_low"] > 0.0


def test_text_release_has_no_author_workstation_paths() -> None:
    windows_absolute = re.compile(r"[A-Za-z]:[\\/](?:Users|AutoResearchClaw)[\\/]")
    mac_user = "/" + "Users/"
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".toml", ".yml", ".yaml", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert windows_absolute.search(text) is None, path
        assert mac_user not in text, path
