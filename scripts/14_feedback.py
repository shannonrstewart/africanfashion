"""Run the community-feedback loop on a form export.

    uv run python scripts/14_feedback.py --input benchmark/feedback/<export>.csv --run sample
    uv run python scripts/14_feedback.py --input benchmark/feedback/synthetic_conflicts.csv --run synthetic --apply

Writes results/feedback/<run>/: submissions_parsed.csv (pseudonymised), weights.csv,
label_decisions.csv, label_overrides.csv, new_data_candidates.csv, expert_queue.csv, run_meta.json.
data/splits/split_v1.csv is never modified; --apply writes split_v2.csv beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark import feedback as fb  # noqa: E402
from benchmark.runmeta import write_run_meta  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--run", required=True, help="output folder name under results/feedback/")
    ap.add_argument("--apply", action="store_true", help="also write data/splits/split_v2.csv")
    ap.add_argument("--max-dist", type=int, default=fb.MATCH_MAX_DIST)
    args = ap.parse_args()

    out = ROOT / "results" / "feedback" / args.run
    out.mkdir(parents=True, exist_ok=True)
    tax = Taxonomy.load()
    manifest = pd.read_csv(ROOT / "data" / "clean" / "manifest.csv")
    H = np.load(ROOT / "data" / "clean" / "hashes.npy")

    subs = fb.parse_submissions(args.input, tax)
    subs = fb.attach_images(subs, ROOT / "data" / "feedback" / "uploads", H, manifest, args.max_dist)
    subs = fb.weigh(subs, tax)

    votes = subs[(subs.image_status == "matched") & subs.proposed_class.notna()]
    decisions = fb.aggregate(votes)
    queue = fb.expert_queue(subs, decisions)
    split = pd.read_csv(ROOT / "data" / "splits" / "split_v1.csv")
    v2, overrides = fb.apply_overrides(split, manifest, decisions)

    private = ["submission_id", "submitter", "created_at", "tradition", "people", "technique", "proposed_class",
               "class_note", "contribution_type", "anonymous", "idx", "cluster", "current_label", "hash_dist",
               "image_status", "upload_sha256", "n_checked", "n_contradictions", "quiz_factor", "weight", "reason"]
    subs[private].to_csv(out / "submissions_parsed.csv", index=False)
    subs[["submission_id", "submitter", "n_checked", "n_contradictions", "consistency_factor", "quiz_factor",
          "weight", "reason"]].to_csv(out / "weights.csv", index=False)
    decisions.to_csv(out / "label_decisions.csv", index=False)
    overrides.to_csv(out / "label_overrides.csv", index=False)
    subs[subs.image_status == "no_match"][["submission_id", "submitter", "tradition", "proposed_class",
                                           "upload_sha256", "hash_dist", "weight"]].to_csv(
        out / "new_data_candidates.csv", index=False)
    queue.to_csv(out / "expert_queue.csv", index=False)
    write_run_meta(out, kind="feedback", params={"input": str(args.input), "max_dist": args.max_dist,
                   "W_EXPERT": fb.W_EXPERT, "W_SOURCE": fb.W_SOURCE, "MIN_SUBMITTERS": fb.MIN_SUBMITTERS,
                   "CONTRADICTION_FACTOR": fb.CONTRADICTION_FACTOR})
    if args.apply:
        v2.to_csv(ROOT / "data" / "splits" / "split_v2.csv", index=False)

    print(subs[["submission_id", "tradition", "image_status", "idx", "hash_dist", "current_label",
                "proposed_class", "weight"]].to_string(index=False))
    print("\nDecisions:\n" + (decisions.drop(columns=["submissions"]).to_string(index=False) if len(decisions) else "  none"))
    print(f"\nLabel overrides: {len(overrides)} | new-data candidates: {int((subs.image_status == 'no_match').sum())} "
          f"| expert queue: {len(queue)} | wrote {out}")


if __name__ == "__main__":
    main()
