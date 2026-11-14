"""Community feedback loop: parse form submissions, weight them, vote on labels.

Pipeline (see benchmark/FEEDBACK.md):
  parse_submissions -> attach_images (pHash match) -> weigh -> aggregate -> expert_queue

Design rules:
  * Trust is based on demonstrated knowledge, not identity. Self-reported credentials are
    kept as description only and never enter the weight.
  * This module never judges cultural claims. It only (a) checks submissions against facts
    the expert has already verified in taxonomy.yaml and (b) counts weighted votes.
    Everything else goes to the expert queue.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .imghash import best_match, hash_bytes
from .taxonomy import Taxonomy, norm_text

# ---- tunables --------------------------------------------------------------------------
# Classes whose labels the expert has verified (taxonomy.yaml: "Everything outside
# Adire/Idoma/Igala/Igbo/Tiv is still an unverified draft").
VERIFIED_CLASSES = frozenset({"Adire", "Idoma", "Igala", "Igbo", "Tiv"})
W_EXPERT = 3.0            # the current label's vote when its class is expert-verified
W_SOURCE = 1.0            # ... when it is not
MIN_SUBMITTERS = 2        # distinct submitters needed to change a label
CONTRADICTION_FACTOR = 0.0
MATCH_MAX_DIST = 6        # pHash Hamming distance (of 64) to call an upload "the same image"

# Form columns we rely on. Others are ignored.
COL_ID = "id"
COL_NAME = "Your name or Preferred Identifier"
COL_TRADITION = "Which cultural textile or style tradition is your submission about?"
COL_KIND = "How would you like to contribute?"
COL_CONNECTION = "What is your connection to this craft or community?"
COL_KNOWLEDGE = "Knowledge and Contribution"
COL_EVIDENCE = "Supporting Evidence"
COL_CREDIT = "How would you like your contribution to be credited?"
COL_CONSENT = "Declaration and Consent"
COL_TIME = "created_at"
COL_QUIZ = "quiz_score"   # optional, 0-1. Not on the form yet.

_TAG = re.compile(r"<[^>]+>")


def strip_html(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = re.sub(r"</p>|<br\s*/?>", " ", str(s))
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", s))).strip()


def pseudonym(name: str) -> str:
    """Stable pseudonymous ID so one person's votes can be de-duplicated without storing the name."""
    return "sub-" + hashlib.sha256(("africanfashion-feedback|" + str(name).strip().lower()).encode()).hexdigest()[:8]


# ---- class lookup ------------------------------------------------------------------------
def class_lookups(tax: Taxonomy):
    """technique -> classes, people -> classes, built from taxonomy.yaml 'classes'."""
    by_tech: dict[str, list[str]] = {}
    by_people: dict[str, list[str]] = {}
    for cname, spec in tax.classes.items():
        for t in spec.get("techniques") or []:
            by_tech.setdefault(t, []).append(cname)
        for p in spec.get("peoples") or []:
            by_people.setdefault(p, []).append(cname)
    return by_tech, by_people


def map_tradition(value: str, tax: Taxonomy) -> dict:
    """'Yoruba — Adire' -> {people, technique, cls, note}. cls is None if unmapped or ambiguous."""
    parts = [p.strip() for p in re.split(r"\s*[—–]\s*|\s+-\s+", str(value or ""), maxsplit=1)]
    people_txt = parts[0] if parts else ""
    tech_txt = parts[1] if len(parts) > 1 else ""
    people = tax.resolve_people(people_txt) if people_txt else None
    technique = None
    for chunk in re.split(r"\s*/\s*|\s*,\s*", tech_txt):   # "Apa / Edema" -> first recognised
        technique = technique or (tax.resolve_technique(chunk) if chunk else None)
    by_tech, by_people = class_lookups(tax)
    cls, note = None, ""
    tc = by_tech.get(technique, []) if technique else []
    pc = by_people.get(people, []) if people else []
    if len(tc) == 1:
        cls = tc[0]
    elif len(pc) == 1:
        cls = pc[0]
    elif tc or pc:
        note = "ambiguous"
    else:
        note = "unmapped"
    if cls and people and technique:  # the tradition field contradicting itself
        if people not in (tax.techniques[technique].get("peoples") or [people]):
            note = "tradition_field_inconsistent"
    return {"people": people, "technique": technique, "cls": cls, "note": note}


# ---- parsing -----------------------------------------------------------------------------
def parse_submissions(csv_path: str | Path, tax: Taxonomy) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows = []
    for _, r in raw.iterrows():
        m = map_tradition(r.get(COL_TRADITION, ""), tax)
        quiz = r.get(COL_QUIZ, "")
        rows.append({
            "submission_id": r[COL_ID],
            "submitter": pseudonym(r.get(COL_NAME, r[COL_ID])),
            "created_at": r.get(COL_TIME, ""),
            "tradition": r.get(COL_TRADITION, ""),
            "people": m["people"], "technique": m["technique"],
            "proposed_class": m["cls"], "class_note": m["note"],
            "contribution_type": r.get(COL_KIND, ""),
            "connection": strip_html(r.get(COL_CONNECTION, "")),   # descriptive only, never weighted
            "knowledge": strip_html(r.get(COL_KNOWLEDGE, "")),
            "evidence": r.get(COL_EVIDENCE, ""),
            "anonymous": "private" in r.get(COL_CREDIT, "").lower() or "anonym" in r.get(COL_CREDIT, "").lower(),
            "consent": str(r.get(COL_CONSENT, "")).strip().upper() == "TRUE",
            "quiz_score": float(quiz) if str(quiz).strip() else np.nan,
        })
    df = pd.DataFrame(rows)
    return df[df.consent].reset_index(drop=True)  # no consent, no use


# ---- image linkage -----------------------------------------------------------------------
def fetch_upload(src: str, cache_dir: str | Path) -> bytes | None:
    """Fetch an upload (URL or local path). Signed URLs expire, so bytes are cached by URL hash."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / (hashlib.sha256(src.encode()).hexdigest()[:24] + ".bin")
    if key.exists():
        return key.read_bytes()
    try:
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "benchmark-feedback"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read(20_000_000)
        else:
            data = Path(src).read_bytes()
    except Exception:
        return None
    key.write_bytes(data)
    return data


def attach_images(subs: pd.DataFrame, cache_dir, H: np.ndarray, manifest: pd.DataFrame,
                  max_dist: int = MATCH_MAX_DIST) -> pd.DataFrame:
    """Match each upload to a dataset image by pHash. Adds idx, cluster, current_label, hash_dist,
    image_status in {matched, no_match, unreadable}."""
    flat = manifest.flat.to_numpy()
    out = []
    for r in subs.itertuples():
        info = {"idx": pd.NA, "cluster": pd.NA, "current_label": None, "hash_dist": pd.NA,
                "upload_sha256": None, "image_status": "unreadable"}
        data = fetch_upload(r.evidence, cache_dir) if r.evidence else None
        if data:
            h = hash_bytes(data)
            info["upload_sha256"] = h["sha256"]
            if not h["flat"]:
                row, dist = best_match(h["hashes"][0], H, flat)
                info["hash_dist"] = dist
                if dist <= max_dist:
                    m = manifest.iloc[row]
                    info.update(idx=int(m.idx), cluster=int(m.cluster), current_label=m.clean_label,
                                image_status="matched")
                else:
                    info["image_status"] = "no_match"
        out.append(info)
    return pd.concat([subs.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ---- weighting ---------------------------------------------------------------------------
def consistency_check(row, tax: Taxonomy) -> dict:
    """Check a submission against expert-verified taxonomy facts only.

    Contradiction = the text (or the tradition field) names a verified technique together with
    peoples, but none of those peoples is one the taxonomy associates with the technique.
    Known limits (documented in FEEDBACK.md): no negation handling, and claims that mention no
    taxonomy entity (history, materials, dates) cannot be checked at all.
    """
    t = norm_text(row.knowledge)
    techs = {c for _, c in tax.tidx["technique"].find_all(t)}
    peoples = {c for _, c in tax.tidx["people"].find_all(t)}
    if row.technique:
        techs.add(row.technique)
    # peoples come from the text only: the tradition field's own people would always "agree"
    by_tech, _ = class_lookups(tax)
    checked, problems = 0, []
    if row.class_note == "tradition_field_inconsistent":
        checked += 1
        problems.append(f"tradition field pairs {row.people} with {row.technique}, which the taxonomy ties to other peoples")
    for tech in sorted(techs):
        if not any(c in VERIFIED_CLASSES for c in by_tech.get(tech, [])):
            continue
        owners = set(tax.techniques[tech].get("peoples") or [])
        if not owners or not peoples:
            continue
        checked += 1
        if not (owners & peoples):
            problems.append(f"{tech} is associated with {'/'.join(sorted(owners))} in the verified taxonomy, "
                            f"but submission names {'/'.join(sorted(peoples))}")
    return {"n_checked": checked, "n_contradictions": len(problems), "details": "; ".join(problems)}


def weigh(subs: pd.DataFrame, tax: Taxonomy) -> pd.DataFrame:
    """weight = consistency_factor x quiz_factor. Credentials are deliberately not used."""
    rows = []
    for r in subs.itertuples():
        c = consistency_check(r, tax)
        cf = CONTRADICTION_FACTOR if c["n_contradictions"] else 1.0
        qf = 1.0 if pd.isna(r.quiz_score) else float(np.clip(r.quiz_score, 0, 1))
        rows.append({"submission_id": r.submission_id, "n_checked": c["n_checked"],
                     "n_contradictions": c["n_contradictions"], "consistency_factor": cf,
                     "quiz_factor": qf, "weight": cf * qf, "reason": c["details"]})
    return subs.merge(pd.DataFrame(rows), on="submission_id")


# ---- voting ------------------------------------------------------------------------------
def aggregate(votes: pd.DataFrame) -> pd.DataFrame:
    """Weighted majority vote per dedup cluster.

    votes needs: submission_id, submitter, created_at, cluster, current_label, proposed_class, weight.
    The current label counts as one vote (W_EXPERT if its class is verified, else W_SOURCE); crowd
    votes for the same class add to it. A label changes only if the leading alternative has
    >= MIN_SUBMITTERS distinct submitters with weight > 0 AND a strictly larger total than the current label.
    One vote per submitter per cluster (their latest).
    """
    v = votes.dropna(subset=["cluster", "proposed_class"]).copy()
    v = v.sort_values("created_at").drop_duplicates(["submitter", "cluster"], keep="last")
    out = []
    for cl, g in v.groupby("cluster"):
        cur = g.current_label.iloc[0]
        totals = {cur: W_EXPERT if cur in VERIFIED_CLASSES else W_SOURCE}
        who: dict[str, set] = {}
        for r in g.itertuples():
            totals[r.proposed_class] = totals.get(r.proposed_class, 0.0) + r.weight
            if r.weight > 0:
                who.setdefault(r.proposed_class, set()).add(r.submitter)
        alts = {k: t for k, t in totals.items() if k != cur and who.get(k)}
        decision, new = "keep", cur
        if alts:
            top = max(alts, key=alts.get)
            if len(who[top]) >= MIN_SUBMITTERS and totals[top] > totals[cur]:
                decision, new = "change", top
            else:
                decision = "contested"
        out.append({"cluster": int(cl), "current_label": cur, "new_label": new, "decision": decision,
                    "current_verified": cur in VERIFIED_CLASSES,
                    "totals": "; ".join(f"{k}={t:g}" for k, t in sorted(totals.items(), key=lambda kv: -kv[1])),
                    "n_votes": len(g), "n_zero_weight": int((g.weight == 0).sum()),
                    "submissions": ",".join(g.submission_id)})
    cols = ["cluster", "current_label", "new_label", "decision", "current_verified", "totals",
            "n_votes", "n_zero_weight", "submissions"]
    return pd.DataFrame(out, columns=cols)


# ---- expert queue ------------------------------------------------------------------------
def expert_queue(subs: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Everything needing human judgement. Listed, not decided."""
    items = []
    contested = set(decisions[decisions.decision.isin(["contested", "change"])].cluster) if len(decisions) else set()
    for r in subs.itertuples():
        reasons = []
        if r.knowledge:
            reasons.append("cultural_claim_text")
        if r.weight == 0:
            reasons.append("zero_weight:" + (r.reason or "no_reason"))
        if r.image_status == "no_match":
            reasons.append("new_data_candidate")
        elif r.image_status == "unreadable":
            reasons.append("upload_unreadable")
        if r.proposed_class is None or pd.isna(r.proposed_class):
            reasons.append(f"tradition_not_in_taxonomy:{r.class_note}")
        elif r.class_note:
            reasons.append(f"tradition:{r.class_note}")
        if pd.notna(r.cluster) and int(r.cluster) in contested:
            reasons.append("label_vote_contested_or_changed")
        if reasons:
            items.append({"submission_id": r.submission_id, "idx": r.idx, "tradition": r.tradition,
                          "current_label": r.current_label, "reasons": " | ".join(reasons),
                          "knowledge": r.knowledge})
    return pd.DataFrame(items)


def apply_overrides(split: pd.DataFrame, manifest: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel every image in a changed cluster that carries the old label. Returns (split_v2, overrides)."""
    chg = decisions[decisions.decision == "change"]
    rows = []
    for d in chg.itertuples():
        mem = manifest[(manifest.cluster == d.cluster) & (manifest.clean_label == d.current_label)]
        for idx in mem.idx:
            rows.append({"idx": int(idx), "cluster": d.cluster, "old_label": d.current_label,
                         "new_label": d.new_label, "evidence": d.totals, "submissions": d.submissions})
    ov = pd.DataFrame(rows, columns=["idx", "cluster", "old_label", "new_label", "evidence", "submissions"])
    v2 = split.copy()
    v2["label_source"] = "v1"
    m = v2.idx.isin(ov.idx)
    v2.loc[m, "clean_label"] = v2.loc[m, "idx"].map(dict(zip(ov.idx, ov.new_label)))
    v2.loc[m, "label_source"] = "community_vote"
    return v2, ov
