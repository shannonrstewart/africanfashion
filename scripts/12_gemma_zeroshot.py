"""Zero-shot evaluation of Gemma 4 (default E4B-it) on the held-out test split, structured protocol.

Same split, prompt file, greedy decoding, and scorer as scripts/07_qwen_zeroshot.py, so the numbers are
directly comparable. Differences that are inherent to the model and are recorded in run_meta.json:
  * image input is a fixed token budget (default 280) instead of Qwen's size-dependent token count;
  * the 2.8B-parameter per-layer-embedding table (a lookup, not compute) is kept in CPU RAM so the
    rest of the model fits on a 16 GB GPU in bf16. Lookups run on CPU; results are moved to the GPU.

Usage:
    uv run python scripts/12_gemma_zeroshot.py --limit 16        # smoke test
    uv run python scripts/12_gemma_zeroshot.py                    # full test split
    uv run python scripts/12_gemma_zeroshot.py --adapter DIR --out results/gemma4_e4b_lora_eval
Resumable: ids already present in the raw output file are skipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

from benchmark.runmeta import write_run_meta  # noqa: E402
from benchmark.scorer import format_table, score, write_outputs  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402

# reuse parse_json / resize so both models see identical preprocessing and parsing
_s = importlib.util.spec_from_file_location("qz", ROOT / "scripts" / "07_qwen_zeroshot.py")
qz = importlib.util.module_from_spec(_s)
_s.loader.exec_module(qz)

MODEL_ID = "google/gemma-4-E4B-it"
PROMPT = ROOT / "benchmark/prompts/structured.txt"
MAX_NEW = 200


def load_model(model_id: str, offload_ple: bool = True):
    """bf16 on GPU, except the per-layer embedding table (and unused audio modules), which stay on CPU."""
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)  # CPU first
    lm = model.model.language_model
    stash = {}
    if offload_ple:
        stash["ple"] = lm.embed_tokens_per_layer
        lm.embed_tokens_per_layer = torch.nn.Identity()
    for name in ("audio_tower", "embed_audio"):
        if hasattr(model.model, name):
            stash[name] = getattr(model.model, name)
            setattr(model.model, name, torch.nn.Identity())
    model.to("cuda")
    for name, mod in stash.items():
        if name == "ple":
            lm.embed_tokens_per_layer = mod
            mod.register_forward_pre_hook(lambda m, a: tuple(x.to("cpu") for x in a))
            mod.register_forward_hook(lambda m, a, out: out.to("cuda"))
        else:
            setattr(model.model, name, mod)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", default=str(ROOT / "results" / "gemma4_e4b_zeroshot"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--image-tokens", type=int, default=280, choices=[70, 140, 280, 560, 1120])
    ap.add_argument("--no-offload", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    te = sp[sp.split == "test"].sort_values("idx")
    if args.limit:
        te = te.groupby("clean_label", group_keys=False).head(max(1, args.limit // te.clean_label.nunique()))
    print(f"test items: {len(te)}")

    ds = load_dataset("inuwamobarak/african-atire", split="train")
    proc = AutoProcessor.from_pretrained(args.model_id)
    proc.tokenizer.padding_side = "left"
    ip = proc.image_processor
    print("image_seq_length (processor default):", getattr(ip, "image_seq_length", None))
    if getattr(ip, "image_seq_length", None) != args.image_tokens:
        ip.image_seq_length = args.image_tokens
        if hasattr(proc, "image_seq_length"):
            proc.image_seq_length = args.image_tokens

    model = load_model(args.model_id, offload_ple=not args.no_offload)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    write_run_meta(out, kind="zeroshot_vlm" if not args.adapter else "finetuned_vlm_eval", model=args.model_id,
                   prompts={"structured": PROMPT},
                   decoding={"do_sample": False, "max_new_tokens": MAX_NEW, "dtype": "bfloat16",
                             "max_image_side_preresize": qz.MAX_SIDE, "image_token_budget": args.image_tokens,
                             "batch": args.batch, "thinking": "off (chat-template default)"},
                   params={"n_items": len(te), "limit": args.limit, "adapter": args.adapter,
                           "per_layer_embeddings_on_cpu": not args.no_offload})
    tax = Taxonomy.load()
    gold = [{"id": str(r.idx), "class": r.clean_label} for r in te.itertuples()]
    prompt = PROMPT.read_text().strip()

    raw_path = out / "structured_raw.jsonl"
    done = set()
    if raw_path.exists():
        done = {json.loads(l)["id"] for l in raw_path.read_text().splitlines() if l.strip()}
    todo = [r for r in te.itertuples() if str(r.idx) not in done]
    print(f"[structured] {len(todo)} to run ({len(done)} cached)")
    t0 = time.time()
    with open(raw_path, "a") as f:
        for b in range(0, len(todo), args.batch):
            rows = todo[b:b + args.batch]
            ims = [qz.resize(ds[int(r.idx)]["image"].convert("RGB")) for r in rows]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in rows]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = proc(text=texts, images=[[im] for im in ims], return_tensors="pt", padding=True).to("cuda")
            if b == 0:
                print("input shape:", tuple(inp["input_ids"].shape), "| keys:", list(inp.keys()))
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
            outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": str(r.idx), "text": o}) + "\n")
            f.flush()
            n = b + len(rows)
            print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/img  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                  flush=True)

    raws = {json.loads(l)["id"]: json.loads(l)["text"] for l in raw_path.read_text().splitlines() if l.strip()}
    preds, n_bad = [], 0
    for g in gold:
        d, ok = qz.parse_json(raws.get(g["id"], ""))
        n_bad += not ok
        preds.append({"id": g["id"], **d})
    with open(out / "structured_preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    metrics, per_item = score(preds, gold, tax)
    write_outputs(metrics, per_item, out / "structured")
    print(f"\n=== structured (unparseable JSON: {n_bad}) ===")
    print(format_table(metrics))


if __name__ == "__main__":
    main()
