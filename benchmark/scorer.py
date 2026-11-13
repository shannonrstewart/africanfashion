"""Granularity-ladder scorer.

Ladder levels:
    L1 Africa     - identifies the item as African
    L2 Region     - West / South / East / North / Central
    L3 People     - prediction is in the item's gold set of peoples
    L4 Technique  - prediction is in the item's gold set of textiles/techniques

Deepest correct rung, per item:
    3 ("specific") if L3 OR L4 is correct; no requirement on L1/L2, since
      techniques can span peoples and regions.
    2 if L1 and L2 are correct
    1 if L1 is correct (the "monolithic Africa" outcome if it stops here)
    0 otherwise

Inputs (JSONL):
  gold:  {"id": ..., "class": "<dataset class>"}  plus optional per-item
         overrides: "african", "region", "peoples", "techniques".
         Control (non-African) items: {"id": ..., "african": false}.
  pred:  {"id": ..., "continent"|"african", "region", "country", "people",
         "technique", "raw"?}  (structured / extracted answers)
     or  {"id": ..., "pred_class": "<dataset class>"}  (closed-set classifiers)

Usage:
  uv run python -m benchmark.scorer --pred P.jsonl --gold G.jsonl --out results/run1
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .taxonomy import DEFAULT_TAXONOMY, Taxonomy, first_value, is_abstain, norm_field

LEVELS = ("l1_africa", "l2_region", "l3_people", "l4_technique")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def _resolve_african(pred: dict) -> bool | None:
    if "african" in pred and pred["african"] is not None:
        v = pred["african"]
        if isinstance(v, bool):
            return v
        s = norm_field(str(v))
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None
    cont = first_value(pred.get("continent"))
    if is_abstain(cont):
        return None
    return "africa" in norm_field(str(cont))


def normalize_prediction(pred: dict, tax: Taxonomy, infer: bool = True) -> dict:
    """Map a raw prediction to canonical ladder values.

    With infer=True (default), facts the answer implies are filled in when not
    stated: naming an African country, people or technique implies "African";
    a country, people or technique implies its region. Explicit statements
    always take precedence, including wrong ones.
    """
    if pred.get("pred_class") is not None:
        cls = pred["pred_class"]
        cls = tax.canonical_class(cls)
        if cls not in tax.classes:
            return {"african": None, "region": None, "people": None, "technique": None,
                    "generic": False, "abstain": {lvl: True for lvl in LEVELS}}
        g = tax.class_gold(cls)
        return {"african": True, "region": g["region"],
                "people": (g["peoples"] or [None])[0],
                "technique": (g["techniques"] or [None])[0],
                "generic": False,
                "abstain": {"l1_africa": False, "l2_region": g["region"] is None,
                            "l3_people": not g["peoples"], "l4_technique": not g["techniques"]}}

    african = _resolve_african(pred)
    region = tax.resolve_region(pred.get("region"))
    country = tax.resolve_country(pred.get("country"))
    people = tax.resolve_people(pred.get("people"))
    technique = tax.resolve_technique(pred.get("technique"))
    generic = tax.has_generic(pred.get("technique"), pred.get("people"))
    if pred.get("generic") is True:
        generic = True

    abstain = {
        "l1_africa": african is None,
        "l2_region": is_abstain(pred.get("region")) and (not infer or country is None),
        "l3_people": is_abstain(pred.get("people")),
        "l4_technique": is_abstain(pred.get("technique")),
    }

    if infer:
        if region is None and is_abstain(pred.get("region")):
            region = (tax.country_region.get(country) if country else None) \
                or (tax.people_region.get(people) if people else None) \
                or (tax.techniques[technique].get("region") if technique else None)
            if region is not None:
                abstain["l2_region"] = False
        if african is None and (region or country or people or technique or generic):
            african = True
            abstain["l1_africa"] = False

    return {"african": african, "region": region, "people": people,
            "technique": technique, "generic": generic, "abstain": abstain}


def gold_for(item: dict, tax: Taxonomy) -> dict:
    if item.get("african") is False:
        return {"african": False, "region": None, "peoples": [], "techniques": [],
                "class": item.get("class") or "__control__"}
    cls = tax.canonical_class(item["class"]) if item.get("class") else None
    g = tax.class_gold(cls) if cls else \
        {"african": True, "region": None, "peoples": [], "techniques": []}
    for k in ("region", "peoples", "techniques"):
        if k in item and item[k] is not None:
            g[k] = item[k]
    g["class"] = cls or "__custom__"
    return g


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_item(p: dict, g: dict) -> dict:
    """Return per-level correctness (True/False, None = not scoreable) and rung."""
    if g["african"] is False:  # control item: only L1 (correct rejection) applies
        return {"l1_africa": p["african"] is False, "l2_region": None,
                "l3_people": None, "l4_technique": None, "rung": None, "control": True}
    l1 = p["african"] is True
    l2 = (p["region"] == g["region"]) if g["region"] else None
    l3 = (p["people"] in g["peoples"]) if g["peoples"] else None
    l4 = (p["technique"] in g["techniques"]) if g["techniques"] else None
    # The rung is only defined when every rung is reachable, i.e. the gold has a
    # region and at least one people or technique. Otherwise a perfect model
    # would be capped at rung 1 and miscounted as "monolithic collapse".
    if not g["region"] or not (g["peoples"] or g["techniques"]):
        rung = None
    elif l3 or l4:
        rung = 3
    elif l1 and l2:
        rung = 2
    elif l1:
        rung = 1
    else:
        rung = 0
    return {"l1_africa": l1, "l2_region": l2, "l3_people": l3, "l4_technique": l4,
            "rung": rung, "control": False}


def _summarise(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    for lvl in LEVELS:
        scored = [r[lvl] for r in rows if r[lvl] is not None]
        out[f"acc_{lvl}"] = (sum(scored) / len(scored)) if scored else None
        out[f"n_{lvl}"] = len(scored)
        out[f"abstain_{lvl}"] = (sum(r[f"abstain_{lvl}"] for r in rows) / len(rows)) if rows else None
    rungs = [r["rung"] for r in rows if r["rung"] is not None]
    out["n_rung_scored"] = len(rungs)
    if rungs:
        c = Counter(rungs)
        out["rung_dist"] = {str(k): c.get(k, 0) / len(rungs) for k in range(4)}
        out["mean_rung"] = sum(rungs) / len(rungs)
        out["monolithic_collapse_rate"] = c.get(1, 0) / len(rungs)
        out["specific_rate"] = c.get(3, 0) / len(rungs)
    out["generic_term_rate"] = (sum(r["generic"] for r in rows) / len(rows)) if rows else None
    return out


def score(preds: list[dict], gold: list[dict], tax: Taxonomy, infer: bool = True) -> tuple[dict, list[dict]]:
    pred_by_id = {str(p["id"]): p for p in preds}
    per_item = []
    n_missing = 0
    for item in gold:
        iid = str(item["id"])
        raw = pred_by_id.get(iid)
        if raw is None:
            n_missing += 1
            raw = {"id": iid}
        p = normalize_prediction(raw, tax, infer=infer)
        g = gold_for(item, tax)
        s = score_item(p, g)
        per_item.append({
            "id": iid, "gold_class": g["class"],
            "pred_african": p["african"], "pred_region": p["region"],
            "pred_people": p["people"], "pred_technique": p["technique"],
            "generic": p["generic"],
            **{lvl: s[lvl] for lvl in LEVELS},
            **{f"abstain_{lvl}": p["abstain"][lvl] for lvl in LEVELS},
            "rung": s["rung"], "control": s["control"],
        })

    african_rows = [r for r in per_item if not r["control"]]
    control_rows = [r for r in per_item if r["control"]]
    metrics = {
        "taxonomy_version": tax.raw.get("version"),
        "taxonomy_status": tax.raw.get("status"),
        "infer": infer,
        "n_gold": len(gold), "n_missing_predictions": n_missing,
        "overall": _summarise(african_rows),
        "per_class": {},
    }
    by_class = defaultdict(list)
    for r in african_rows:
        by_class[r["gold_class"]].append(r)
    for cls in sorted(by_class):
        metrics["per_class"][cls] = _summarise(by_class[cls])
    if control_rows:
        metrics["controls"] = {
            "n": len(control_rows),
            "correct_rejection_rate": sum(bool(r["l1_africa"]) for r in control_rows) / len(control_rows),
            "false_africa_rate": sum(r["pred_african"] is True for r in control_rows) / len(control_rows),
        }
    return metrics, per_item


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_outputs(metrics: dict, per_item: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if per_item:
        with open(out / "per_item.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_item[0]))
            w.writeheader()
            w.writerows(per_item)


def format_table(metrics: dict) -> str:
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"

    head = f"{'class':<20}{'n':>6}  {'L1':>5} {'L2':>5} {'L3':>5} {'L4':>5}  {'r0':>5} {'r1':>5} {'r2':>5} {'r3':>5}  {'mean':>5}"
    lines = [head, "-" * len(head)]
    rows = list(metrics["per_class"].items()) + [("OVERALL", metrics["overall"])]
    for name, m in rows:
        rd = m.get("rung_dist", {})
        lines.append(
            f"{name:<20}{m['n']:>6}  "
            + " ".join(pct(m[f"acc_{lvl}"]) for lvl in LEVELS) + "  "
            + " ".join(pct(rd.get(str(k))) for k in range(4))
            + ("    n/a" if m.get("mean_rung") is None else f"  {m['mean_rung']:5.2f}")
        )
    o = metrics["overall"]
    lines.append("")
    if o["n_rung_scored"] < o["n"]:
        lines.append(f"Note: rungs computed for {o['n_rung_scored']}/{o['n']} items; the rest "
                     "have incomplete gold labels in the taxonomy (n/a).")
    lines.append(f"Monolithic-collapse rate (stops at 'African'): {pct(o.get('monolithic_collapse_rate'))}%")
    lines.append(f"Specific rate (people or technique correct):   {pct(o.get('specific_rate'))}%")
    lines.append(f"Generic-term rate (ankara, wax print, ...):    {pct(o.get('generic_term_rate'))}%")
    if "controls" in metrics:
        lines.append(f"Controls: correct rejection {pct(metrics['controls']['correct_rejection_rate'])}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-infer", action="store_true", help="Do not infer African/region from more specific answers")
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    metrics, per_item = score(read_jsonl(args.pred), read_jsonl(args.gold), tax, infer=not args.no_infer)
    write_outputs(metrics, per_item, args.out)
    print(format_table(metrics))
    print(f"\nWrote {args.out}/metrics.json and per_item.csv")


if __name__ == "__main__":
    main()
