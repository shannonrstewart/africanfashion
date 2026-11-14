"""LoRA fine-tune of Qwen3-VL-4B-Instruct on the train split, using the structured-JSON protocol.

The training prompt is the SAME file the zero-shot run used (benchmark/prompts/structured.txt), and
the target is the class's gold answer from benchmark/taxonomy.yaml rendered as that prompt's JSON.
Evaluation afterwards is scripts/07_qwen_zeroshot.py --adapter <dir>, so before/after numbers come
from identical code, split, prompt and decoding.

Target construction (DRAFT, see notes in run_meta): for fields where the taxonomy lists several
values (countries, peoples, techniques) the FIRST listed value is used. The scorer credits any
value in the gold set, so this only affects what the model is taught to say, not how it is scored.
Taxonomy entries still marked TODO(expert) are therefore baked into the training target.

Usage:
    uv run python scripts/10_qwen_finetune.py --max-steps 5 --out results/qwen3vl_4b_smoke_ft   # smoke
    uv run python scripts/10_qwen_finetune.py                                                  # full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup  # noqa: E402

from benchmark.runmeta import write_run_meta  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_SIDE = 768
PROMPT_FILE = ROOT / "benchmark/prompts/structured.txt"


def resize(im):
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)))
    return im


def target_json(tax: Taxonomy, cls: str) -> str:
    c = tax.classes[tax.canonical_class(cls)]
    first = lambda k: (c.get(k) or [None])[0]  # noqa: E731
    return json.dumps({
        "continent": "Africa",
        "region": f"{c['region']} Africa",
        "country": first("countries"),
        "people": first("peoples"),
        "technique": first("techniques"),
        "confidence": 1.0,
    }, ensure_ascii=False)


class Collator:
    def __init__(self, proc, prompt, targets, ds):
        self.proc, self.prompt, self.targets, self.ds = proc, prompt, targets, ds
        # token ids of "<|im_start|>assistant\n": the supervised span starts right after the last one
        self.marker = proc.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, rows):
        ims = [resize(self.ds[int(r.idx)]["image"].convert("RGB")) for r in rows]
        texts = []
        for r in rows:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": self.targets[r.clean_label]}]}]
            texts.append(self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        enc = self.proc(text=texts, images=ims, return_tensors="pt", padding=True)
        labels = torch.full_like(enc["input_ids"], -100)
        m = len(self.marker)
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = max(j for j in range(len(ids) - m + 1) if ids[j:j + m] == self.marker) + m
            labels[i, start:] = enc["input_ids"][i, start:]
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "qwen3vl_4b_lora"))
    ap.add_argument("--adapter-dir", default=str(ROOT / "data" / "models" / "qwen3vl_4b_lora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-test cap on optimizer steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-limit", type=int, default=300, help="val items used for the loss curve")
    args = ap.parse_args()
    out, adir = Path(args.out), Path(args.adapter_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    tr = sp[sp.split == "train"].reset_index(drop=True)
    va = sp[sp.split == "val"].sample(frac=1, random_state=args.seed).head(args.val_limit)
    assert not (set(tr.idx) & set(sp[sp.split == "test"].idx)), "train/test overlap"

    tax = Taxonomy.load()
    targets = {c: target_json(tax, c) for c in sorted(tr.clean_label.unique())}
    for c, t in targets.items():
        print(f"{c:20s} -> {t}")
    prompt = PROMPT_FILE.read_text().strip()

    ds = load_dataset("inuwamobarak/african-atire", split="train")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # LoRA on the language model's linear layers only; vision tower stays frozen.
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    coll = Collator(proc, prompt, targets, ds)
    steps_per_epoch = math.ceil(len(tr) / (args.batch * args.accum))
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)

    write_run_meta(out, kind="finetune_vlm_lora", model=MODEL_ID, prompts={"structured": PROMPT_FILE},
                   decoding={"dtype": "bfloat16", "max_image_side": MAX_SIDE},
                   params={"epochs": args.epochs, "max_steps": args.max_steps, "total_steps": total,
                           "batch": args.batch, "accum": args.accum, "lr": args.lr, "lora_rank": args.rank,
                           "n_train": len(tr), "seed": args.seed, "targets": targets,
                           "adapter_dir": str(adir.relative_to(ROOT)) if adir.is_relative_to(ROOT) else str(adir)},
                   notes="Targets use the FIRST listed country/people/technique per class from the draft "
                         "taxonomy; confidence fixed at 1.0. Taxonomy TODO(expert) entries are baked in.")

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for b in range(0, len(va), args.batch):
                batch = coll(list(va.iloc[b:b + args.batch].itertuples())).to("cuda")
                tot += model(**batch).loss.item() * len(batch["input_ids"])
                n += len(batch["input_ids"])
        model.train()
        return tot / n

    log = []
    step, t0 = 0, time.time()
    model.train()
    done = False
    epoch = 0
    while not done:
        order = tr.sample(frac=1, random_state=args.seed + epoch).itertuples()
        order = list(order)
        epoch += 1
        for mb in range(0, len(order), args.batch * args.accum):
            window = order[mb:mb + args.batch * args.accum]
            loss_acc = 0.0
            for b in range(0, len(window), args.batch):
                batch = coll(window[b:b + args.batch]).to("cuda")
                loss = model(**batch).loss * (len(batch["input_ids"]) / len(window))
                loss.backward()
                loss_acc += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": loss_acc,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
            if step % 20 == 0 or step == total or step == steps_per_epoch:
                rec["val_loss"] = val_loss()
            log.append(rec)
            print(rec, flush=True)
            if step >= total:
                done = True
                break
    adir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adir)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"adapter saved to {adir}")


if __name__ == "__main__":
    main()
