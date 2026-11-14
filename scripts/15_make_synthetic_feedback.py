"""Generate benchmark/feedback/synthetic_conflicts.csv: SYNTHETIC form submissions in the real form's
schema, built to exercise the weighted vote. Not real feedback. Upload files go to data/feedback/synthetic_uploads/.
Cases (A-E) are described in benchmark/FEEDBACK.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark import feedback as fb  # noqa: E402

REAL = ROOT / "benchmark" / "feedback" / "community-cultural-knowledge-verification-submissions.csv"
OUT = ROOT / "benchmark" / "feedback" / "synthetic_conflicts.csv"
UP = ROOT / "data" / "feedback" / "synthetic_uploads"
REVIEW = ROOT / "data" / "expert_review" / "images"

def review_image(af_id: str) -> Path:
    return next(REVIEW.glob(f"{af_id}_*.jpg"))


def main() -> None:
    manifest = pd.read_csv(ROOT / "data" / "expert_review" / "manifest.csv")
    pick = lambda label: manifest[manifest.label == label].id.iloc[0]  # noqa: E731
    UP.mkdir(parents=True, exist_ok=True)
    tmpl = pd.read_csv(REAL, dtype=str, keep_default_na=False).iloc[0].to_dict()

    # image-per-case: A and C/D use verified or unverified classes as described in FEEDBACK.md
    img_a = pick("Zulu")             # unverified label
    img_b = manifest[manifest.label == "Adire"].id.iloc[1]    # verified label
    img_c = manifest[manifest.label == "Tiv"].id.iloc[1]      # verified label
    img_d = manifest[manifest.label == "Igala"].id.iloc[1]    # verified label
    rng = np.random.default_rng(0)
    new_img = UP / "E_not_in_dataset.png"
    Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(new_img)

    ok = "Handloom woven cotton cloth, commonly worn at ceremonies."
    plan = [
        # case, image path, tradition, text, n
        ("A", review_image(img_a), "Xhosa — Umbhaco", ok, 3),                 # 3 consistent vs unverified label
        ("B", review_image(img_b), "Igbo — Isi Agu", ok, 2),                  # 2 vs verified label (3.0)
        ("C", review_image(img_c), "Igbo — Isi Agu", ok, 4),                  # 4 vs verified label
        ("D", review_image(img_d), "Idoma — Apa / Edema", ok, 3),             # 3 consistent...
        ("D", review_image(img_d), "Idoma — Apa / Edema",
         "Apa is a Yoruba textile made by palace drummers.", 1),              # ...+1 contradicting (weight 0)
        ("E", new_img, "Igbo — Isi Agu", ok, 1),                              # not in dataset
    ]
    rows, n = [], 0
    for case, path, trad, text, count in plan:
        for k in range(count):
            n += 1
            r = dict(tmpl)
            r.update({"id": f"SYN{case}{n:02d}", fb.COL_NAME: f"SYNTHETIC {case}-{n}",
                      fb.COL_TRADITION: trad, fb.COL_KNOWLEDGE: f"<p>{text}</p>",
                      fb.COL_EVIDENCE: str(path.relative_to(ROOT)), fb.COL_CONNECTION: "<p>SYNTHETIC</p>",
                      fb.COL_TIME: f"2026-11-14 20:{n:02d}"})
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(rows)} synthetic rows)")


if __name__ == "__main__":
    main()
