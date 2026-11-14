"""Zero-shot evaluation of Qwen3-VL on the held-out test split, both prompt protocols.

Usage:
    uv run python scripts/07_qwen_zeroshot.py --limit 20          # smoke test
    uv run python scripts/07_qwen_zeroshot.py                      # full test split
Resumable: ids already present in the raw output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from benchmark.extract import extract_from_text  # noqa: E402
from benchmark.runmeta import write_run_meta  # noqa: E402
from benchmark.scorer import format_table, score, write_outputs  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_SIDE = 768
PROMPTS = {"open": ROOT / "benchmark/prompts/open_ended.txt",
           "structured": ROOT / "benchmark/prompts/structured.txt"}
MAX_NEW = {"open": 300, "structured": 200}


def parse_json(text: str) -> tuple[dict, bool]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d, True
        except json.JSONDecodeError:
            pass
    return {}, False


def resize(im):
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--protocols", nargs="+", default=["structured", "open"])
    ap.add_argument("--out", default=str(ROOT / "results" / "qwen3vl_4b_zeroshot"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-file", default=None, help="override prompt for a diagnostic run (protocol name 'custom')")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to merge in (fine-tuned evaluation)")
    ap.add_argument("--ids-file", default=None, help="text file of dataset idx values to restrict to")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    te = sp[sp.split == "test"].sort_values("idx")
    if args.limit:  # stratified-ish smoke sample: round-robin over classes
        te = te.groupby("clean_label", group_keys=False).head(max(1, args.limit // te.clean_label.nunique()))
    if args.ids_file:
        keep = {int(x) for x in Path(args.ids_file).read_text().split()}
        te = te[te.idx.isin(keep)]
    if args.prompt_file:
        PROMPTS["custom"] = Path(args.prompt_file)
        MAX_NEW["custom"] = 200
        args.protocols = ["custom"]
    print(f"test items: {len(te)}")

    ds = load_dataset("inuwamobarak/african-atire", split="train")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    write_run_meta(out, kind="zeroshot_vlm", model=MODEL_ID,
                   prompts={p: PROMPTS[p] for p in args.protocols},
                   decoding={"do_sample": False, "max_new_tokens": {p: MAX_NEW[p] for p in args.protocols},
                             "dtype": "bfloat16", "max_image_side": MAX_SIDE, "batch": args.batch},
                   params={"n_items": len(te), "ids_file": args.ids_file, "limit": args.limit,
                           "adapter": args.adapter})
    tax = Taxonomy.load()
    gold = [{"id": str(r.idx), "class": r.clean_label} for r in te.itertuples()]

    for proto in args.protocols:
        prompt = PROMPTS[proto].read_text().strip()
        raw_path = out / f"{proto}_raw.jsonl"
        done = set()
        if raw_path.exists():
            done = {json.loads(l)["id"] for l in raw_path.read_text().splitlines() if l.strip()}
        todo = [r for r in te.itertuples() if str(r.idx) not in done]
        print(f"[{proto}] {len(todo)} to run ({len(done)} cached)")
        t0 = time.time()
        with open(raw_path, "a") as f:
            for b in range(0, len(todo), args.batch):
                rows = todo[b:b + args.batch]
                ims = [resize(ds[int(r.idx)]["image"].convert("RGB")) for r in rows]
                msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                        for _ in rows]
                texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
                inp = proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inp, max_new_tokens=MAX_NEW[proto], do_sample=False)
                outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                for r, o in zip(rows, outs):
                    f.write(json.dumps({"id": str(r.idx), "text": o}) + "\n")
                f.flush()
                done_n = b + len(rows)
                print(f"  {done_n}/{len(todo)}  {(time.time() - t0) / done_n:.2f}s/img", flush=True)

        # ---- score
        raws = {json.loads(l)["id"]: json.loads(l)["text"] for l in raw_path.read_text().splitlines() if l.strip()}
        preds, n_bad = [], 0
        for g in gold:
            text = raws.get(g["id"], "")
            if proto == "structured":
                d, ok = parse_json(text)
                n_bad += not ok
                preds.append({"id": g["id"], **d})
            else:
                preds.append({"id": g["id"], **extract_from_text(text, tax)})
        with open(out / f"{proto}_preds.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        metrics, per_item = score(preds, gold, tax)
        write_outputs(metrics, per_item, out / proto)
        print(f"\n=== {proto} (unparseable JSON: {n_bad}) ===")
        print(format_table(metrics))


if __name__ == "__main__":
    main()
