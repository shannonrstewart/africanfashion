"""Linear probe on frozen DINOv2 features, evaluated with the granularity-ladder benchmark.

Model selection uses ONLY the validation split (feature set x regularisation grid); the test
split is evaluated once, with the chosen setting. Reported subsets: all test images, small
images (<128 px short side), larger images, and the expert-review images that fall in test.

Usage:
    uv run python scripts/05_dinov2_probe.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from benchmark.scorer import format_table, score, write_outputs  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402

EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_CSV = ROOT / "data" / "splits" / "split_v1.csv"
OUT_DIR = ROOT / "results" / "dinov2_probe"
MODEL_DIR = ROOT / "data" / "models"
FEATURES = {"cls": slice(0, 768), "cls+patchmean": slice(0, 1536)}
C_GRID = [0.001, 0.01, 0.1, 1.0]
N_BOOT = 1000


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    return sc, clf


def bootstrap_ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[i], p[i]))
        f1.append(f1_score(y[i], p[i], average="macro"))
    return np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    E = np.load(EMB_DIR / "dinov2-base_sq224.npy")
    emb_idx = np.load(EMB_DIR / "dinov2-base_sq224_idx.npy")
    sp = pd.read_csv(SPLIT_CSV)
    pos = pd.Series(np.arange(len(emb_idx)), index=emb_idx)
    sp["row"] = pos[sp.idx].to_numpy()
    tr, va, te = (sp[sp.split == s] for s in ("train", "val", "test"))
    y_tr, y_va, y_te = (d.clean_label.to_numpy() for d in (tr, va, te))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    # ---- model selection on validation only
    grid = []
    for fname, sl in FEATURES.items():
        for C in C_GRID:
            sc, clf = fit(E[tr.row][:, sl], y_tr, C)
            p = clf.predict(sc.transform(E[va.row][:, sl]))
            grid.append({"features": fname, "C": C, "val_acc": accuracy_score(y_va, p),
                         "val_macro_f1": f1_score(y_va, p, average="macro")})
            print(f"  {fname:14s} C={C:<6} val acc {grid[-1]['val_acc']:.3f}  macro-F1 {grid[-1]['val_macro_f1']:.3f}")
    grid = pd.DataFrame(grid)
    best = grid.sort_values("val_macro_f1", ascending=False).iloc[0]
    print(f"\nchosen on validation: {best.features}, C={best.C}")

    # ---- final fit on train, evaluate test once
    sl = FEATURES[best.features]
    sc, clf = fit(E[tr.row][:, sl], y_tr, best.C)
    joblib.dump({"scaler": sc, "clf": clf, "features": best.features, "C": best.C}, MODEL_DIR / "dinov2_probe.joblib")
    proba = clf.predict_proba(sc.transform(E[te.row][:, sl]))
    pred = clf.classes_[proba.argmax(1)]
    te = te.assign(pred=pred, confidence=proba.max(1))
    te.drop(columns=["row"]).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    classes = list(clf.classes_)
    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
    (acc_lo, acc_hi), (f1_lo, f1_hi) = bootstrap_ci(y_te, pred)
    rep = classification_report(y_te, pred, labels=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(rep).T.loc[classes + ["macro avg"], ["precision", "recall", "f1-score", "support"]]

    # confusion matrix (row-normalised)
    cm = confusion_matrix(y_te, pred, labels=classes)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("DINOv2 linear probe: test (row-normalised)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    # ---- ladder benchmark on subsets
    tax = Taxonomy.load()
    subsets = {
        "test (all)": te,
        "test: small (<128 px)": te[te.small],
        "test: >=128 px": te[~te.small],
        "test: expert-review images": te[te.in_expert_sample],
    }
    sub_lines, ladder, sub_stats = [], {}, {}
    for name, d in subsets.items():
        if d.empty:
            continue
        gold = [{"id": str(r.idx), "class": r.clean_label} for r in d.itertuples()]
        preds = [{"id": str(r.idx), "pred_class": r.pred} for r in d.itertuples()]
        metrics, per_item = score(preds, gold, tax)
        ladder[name] = metrics
        sub_stats[name] = {"n": len(d), "accuracy": accuracy_score(d.clean_label, d.pred),
                           "macro_f1": f1_score(d.clean_label, d.pred, average="macro")}
        if name == "test (all)":
            write_outputs(metrics, per_item, OUT_DIR / "ladder")
        sub_lines.append(f"| {name} | {len(d)} | {accuracy_score(d.clean_label, d.pred):.3f} | "
                         f"{f1_score(d.clean_label, d.pred, average='macro'):.3f} | "
                         f"{100 * metrics['overall']['specific_rate']:.1f} | "
                         f"{100 * metrics['overall']['wrong_specific_rate']:.1f} | "
                         f"{100 * metrics['overall']['monolithic_collapse_rate']:.1f} |")
    (OUT_DIR / "ladder_by_subset.json").write_text(json.dumps(ladder, indent=2))
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "model": "facebook/dinov2-base", "features": best.features, "C": float(best.C),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "val_grid": grid.to_dict("records"),
        "test": {"accuracy": acc, "accuracy_ci95": [float(acc_lo), float(acc_hi)],
                 "macro_f1": f1, "macro_f1_ci95": [float(f1_lo), float(f1_hi)]},
        "per_class": {c: {"precision": float(per_class.loc[c, "precision"]), "recall": float(per_class.loc[c, "recall"]),
                          "f1": float(per_class.loc[c, "f1-score"]), "support": int(per_class.loc[c, "support"])}
                      for c in classes},
        "confusion": {"classes": classes, "counts": cm.tolist()},
        "subsets": sub_stats,
    }, indent=2))

    md = ["# DINOv2 linear probe\n",
          f"Frozen `facebook/dinov2-base`, features `{best.features}`, logistic regression C={best.C} "
          f"(class-balanced), chosen on validation macro-F1.\n",
          "## Validation grid\n", grid.round(4).to_markdown(index=False), "",
          "## Test results\n",
          f"- accuracy **{acc:.3f}** (95% CI {acc_lo:.3f}-{acc_hi:.3f}), macro-F1 **{f1:.3f}** "
          f"(95% CI {f1_lo:.3f}-{f1_hi:.3f}), n={len(te)}\n",
          per_class.round(3).to_markdown(), "",
          "## By subset\n",
          "| subset | n | accuracy | macro-F1 | ladder: specific % | wrong-specific % | collapse % |",
          "|---|---:|---:|---:|---:|---:|---:|", *sub_lines, "",
          "## Granularity ladder, all test images\n", "```", format_table(ladder["test (all)"]), "```", "",
          "_A closed-set classifier always answers with one of the 8 classes, so L1 is 100% by "
          "construction and its collapse rate is 0%; its rung-1 answers are all wrong-specific "
          "(e.g. a Zulu image called Igbo). These numbers are the reference point for the VLM "
          "comparison, which has to produce answers open-endedly._",
          "_Gold for Tswana-Shweshwe, Xhosa and Zulu is still a draft awaiting expert verification._"]
    (OUT_DIR / "report.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
