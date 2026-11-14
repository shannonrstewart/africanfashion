# Community feedback loop (basic demonstration)

How form submissions could correct labels or add benchmark data. It is a working sketch with known
gaps (below), not a finished moderation system. Code: `benchmark/feedback.py`, `scripts/14_feedback.py`.

    uv run python scripts/14_feedback.py --input <form export>.csv --run <name> [--apply]

## Pipeline
1. **Parse.** Strip HTML, split the tradition field ("Yoruba — Adire") into people and textile, and map it
   to a class through `taxonomy.yaml`. Names become pseudonyms (`sub-xxxxxxxx`); rows without consent are dropped.
2. **Link to an image.** The form has no image-ID field, so the upload is downloaded (signed links expire,
   so it is cached) and matched to the dataset by perceptual hash (`benchmark/imghash.py`, the same hash as the
   dedup step; rotated/flipped copies match). A match gives the image's dedup cluster, so a vote covers
   near-duplicates. No match means a new-data candidate.
3. **Weight.** `weight = consistency_factor x quiz_factor`.
   - `consistency_factor` is 0 if the submission contradicts a fact the expert has verified in the taxonomy
     (the text pairs a verified textile with a people the taxonomy does not associate with it), else 1.
   - `quiz_factor` is 1 until the form gains a knowledge-question block; it then reads an optional `quiz_score` (0-1) column.
   - Self-reported credentials ("Culture Expert", "Lawyer") are recorded but **never** weighted. Both bad-faith test rows
     claim expertise.
4. **Vote.** Per cluster, the current label counts as one vote (3.0 if its class is expert-verified, else 1.0). Submissions add
   their weight to the class they propose, one vote per submitter. The label changes only if the leading alternative has at
   least 2 distinct submitters with weight > 0 and a strictly larger total than the current label. Otherwise it is marked
   `contested`. Constants are at the top of `feedback.py`.
5. **Output** (`results/feedback/<run>/`): parsed and weighted submissions, `label_decisions.csv`, `label_overrides.csv`,
   `new_data_candidates.csv`, `expert_queue.csv`. `--apply` writes `data/splits/split_v2.csv` (v1 plus overrides, with a
   `label_source` column); `split_v1` is never edited, so earlier scores stay comparable.
6. **Expert queue.** Every free-text cultural claim, zero-weight submission, contested or changed label, and new tradition
   is listed for the expert. The code does not judge these.

## Results
- **Real sample (5 rows):** all five uploads matched their dataset image at hash distance 0; no label changes (no two rows
  concern the same image). The Tiv/A'nger row that claims A'nger is Yoruba got weight 0. The Adire "Indian Kalamkari" bad-faith
  row got weight 1: it names no taxonomy entity, so it can't be checked automatically.
- **Synthetic set** (`benchmark/feedback/synthetic_conflicts.csv`, regenerate with `scripts/15_make_synthetic_feedback.py`;
  every row is marked SYNTHETIC):

| case | setup | outcome |
|---|---|---|
| A | 3 consistent submitters vs an unverified label | changed |
| B | 2 submitters vs a verified label (3.0) | kept, contested |
| C | 4 submitters vs a verified label | changed |
| D | 3 consistent + 1 contradicting (weight 0) vs a verified label | kept, contested (3 vs 3) |
| E | image not in the dataset | new-data candidate |

A change relabels every image in the cluster that carries the old label, so one decision can touch several rows.

## Known gaps
- **No quiz block yet**, so weights only distinguish "contradicts verified facts" from "doesn't". This is the largest gap.
- **Undetectable bad faith:** a false claim that names no taxonomy entity (history, dyes, dates) passes the check. Negation is not
  handled ("not Yoruba" reads as Yoruba). Only the five expert-verified classes can be checked.
- **Form fields:** no image ID, no "proposed label" field, and "contribution type" doesn't match content reliably, so every row
  with a mapped tradition counts as a vote.
- **Sybil attacks:** one person can submit under several names. Pseudonyms only de-duplicate identical names.
- **Overturning expert labels** by crowd majority is possible by design (4 vs 3.0). Any such change should still be confirmed
  by the expert before the benchmark is re-released.
- **New data** is only listed. Adding it also needs a duplicate check against the full set, a label, and rights/consent review.
- Uploads are untrusted files: they are opened with PIL only and never executed.

## Suggested form changes
An explicit image-ID field (or a QR/link that pre-fills it), a "proposed label" dropdown of canonical classes, and a block of
cultural-knowledge questions per tradition whose score fills `quiz_score`.
