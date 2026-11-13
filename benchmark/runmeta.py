"""Write a run_meta.json that pins down exactly what produced a results folder.

Records: model, prompt-file hashes, decoding settings, split and taxonomy hashes (plus the
taxonomy's draft/verified status), scorer/extractor code hashes, library versions, git commit
and dirty flag, and the command line. Hashes (not just names) are what make "which prompt
produced this?" answerable later, even after files are edited.

    from benchmark.runmeta import write_run_meta
    write_run_meta(out_dir, kind="zeroshot", model="Qwen/...", prompts={"structured": path},
                   decoding={...}, params={...})
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "splits" / "split_v1.csv"
TAXONOMY = ROOT / "benchmark" / "taxonomy.yaml"
CODE = {"scorer": ROOT / "benchmark" / "scorer.py", "extractor": ROOT / "benchmark" / "extract.py",
        "taxonomy_loader": ROOT / "benchmark" / "taxonomy.py"}


def sha256(path: str | Path | None) -> str | None:
    p = Path(path) if path else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "scikit-learn"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            pass
    return out


def write_run_meta(out_dir: str | Path, *, kind: str, model: str | None = None,
                   prompts: dict[str, str | Path] | None = None, decoding: dict | None = None,
                   params: dict | None = None, notes: str | None = None,
                   reconstructed: bool = False, filename: str = "run_meta.json") -> Path:
    """reconstructed=True marks metadata filled in after the fact (hashes reflect the files
    as they are NOW, not necessarily as they were when the run happened)."""
    tax = yaml.safe_load(TAXONOMY.read_text())
    status = _git("status", "--porcelain")
    meta = {
        "kind": kind,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reconstructed": reconstructed,
        "model": model,
        "prompts": {k: {"path": str(Path(v).relative_to(ROOT)) if Path(v).is_relative_to(ROOT) else str(v),
                        "sha256": sha256(v)} for k, v in (prompts or {}).items()},
        "decoding": decoding or {},
        "params": params or {},
        "split": {"file": str(SPLIT.relative_to(ROOT)), "sha256": sha256(SPLIT)},
        "taxonomy": {"version": tax.get("version"), "status": tax.get("status"), "sha256": sha256(TAXONOMY)},
        "code_sha256": {k: sha256(v) for k, v in CODE.items()},
        "libraries": _versions(),
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None},
        "argv": sys.argv,
        "notes": notes,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / filename).write_text(json.dumps(meta, indent=2))
    return out / filename
