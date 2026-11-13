"""Export test images and build a browsable HTML review sheet for a zero-shot run.

Each card shows the image (id = dataset row index), gold class, and for each protocol the
model's raw answer plus the ladder rung it received. Filter by class, protocol and rung.

Usage:
    uv run python scripts/08_review_sheet.py [--run results/qwen3vl_4b_zeroshot]
Output: <run>/review/index.html (open in a browser) and <run>/review/images/<idx>_<label>.jpg
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

MAX_SIDE = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "qwen3vl_4b_zeroshot"))
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "review"
    (out / "images").mkdir(parents=True, exist_ok=True)

    te = pd.read_csv(ROOT / "data/splits/split_v1.csv").query("split == 'test'").sort_values("idx")
    protos = [p for p in ("structured", "open") if (run / f"{p}_raw.jsonl").exists()]
    raw, rung = {}, {}
    for p in protos:
        raw[p] = {}
        for line in (run / f"{p}_raw.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                raw[p][d["id"]] = d["text"]
        pi = run / p / "per_item.csv"
        rung[p] = pd.read_csv(pi, dtype={"id": str}).set_index("id") if pi.exists() else None

    ds = load_dataset("inuwamobarak/african-atire", split="train")
    cards = []
    for r in te.itertuples():
        iid = str(r.idx)
        if not any(iid in raw[p] for p in protos):
            continue
        fn = f"images/{iid}_{r.clean_label}.jpg"
        path = out / fn
        if not path.exists():
            im = ds[int(r.idx)]["image"].convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)))
            im.save(path, quality=90)
        body, rungs = [], {}
        for p in protos:
            t = raw[p].get(iid)
            if t is None:
                continue
            rg = None
            if rung[p] is not None and iid in rung[p].index:
                v = rung[p].loc[iid, "rung"]
                rg = None if pd.isna(v) else int(v)
            rungs[p] = rg
            body.append(f'<div class="p"><b>{p}</b> · rung {rg if rg is not None else "n/a"}'
                        f'<pre>{html.escape(t)}</pre></div>')
        cards.append(
            f'<div class="card" data-cls="{html.escape(r.clean_label)}" '
            f'data-rung-structured="{rungs.get("structured", "")}" data-rung-open="{rungs.get("open", "")}">'
            f'<img loading="lazy" src="{fn}"><div class="m"><b>id {iid}</b> · gold: {html.escape(r.clean_label)}'
            f'{" · small" if r.small else ""}</div>{"".join(body)}</div>')

    classes = sorted(te.clean_label.unique())
    page = f"""<!doctype html><meta charset="utf-8"><title>Zero-shot review</title>
<style>body{{font:14px system-ui;margin:16px;background:#fafafa}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;margin:8px 0;display:flex;gap:12px}}
.card img{{max-width:260px;max-height:260px;object-fit:contain;flex:none}}
.m{{margin-bottom:4px}} .p{{margin-bottom:6px}} pre{{white-space:pre-wrap;margin:2px 0;font-size:12px;max-height:200px;overflow:auto}}
.bar{{position:sticky;top:0;background:#fafafa;padding:6px 0}}</style>
<div class="bar">class <select id="c"><option value="">all</option>{"".join(f"<option>{html.escape(c)}</option>" for c in classes)}</select>
protocol <select id="pr">{"".join(f"<option>{p}</option>" for p in protos)}</select>
rung <select id="rg"><option value="">all</option><option>0</option><option>1</option><option>2</option><option>3</option></select>
<span id="n"></span></div>
<div id="list">{"".join(cards)}</div>
<script>
const f=()=>{{const c=cls.value,p=pr.value,r=rg.value;let n=0;
for(const e of document.querySelectorAll('.card')){{const ok=(!c||e.dataset.cls===c)&&(!r||e.dataset['rung'+p[0].toUpperCase()+p.slice(1)]===r);
e.style.display=ok?'':'none';n+=ok}}document.getElementById('n').textContent=n+' shown'}};
const cls=document.getElementById('c'),pr=document.getElementById('pr'),rg=document.getElementById('rg');
[cls,pr,rg].forEach(x=>x.onchange=f);f();
</script>"""
    (out / "index.html").write_text(page)
    print(f"{len(cards)} cards -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
