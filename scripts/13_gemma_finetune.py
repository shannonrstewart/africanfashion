"""LoRA fine-tune of Gemma 4 (default E4B-it) on the train split, structured-JSON protocol.

Mirrors scripts/10_qwen_finetune.py (same split, prompt file, targets, LoRA rank, lr, batch, epochs) so the
two fine-tunes are comparable. Added for long runs on this machine:
  * checkpoint every --ckpt-every optimizer steps (adapter + optimizer + scheduler + position);
    `--resume` continues from the latest checkpoint at the exact data position;
  * early stopping on validation loss: stop when val loss has not improved by --min-delta for
    --patience consecutive validations (a cosine schedule only hits lr=0 on its last step, so lr itself
    is not a usable stop signal); the best-val adapter is saved to <adapter-dir>/best and the last one to
    <adapter-dir>/final;
  * --selfcheck: checks that the loss does not depend on use_cache (Gemma 4 E-models share KV across
    layers; use_cache=False is reported to break them) and exits.

Usage:
    uv run python scripts/13_gemma_finetune.py --selfcheck
    uv run python scripts/13_gemma_finetune.py --max-steps 3 --out SCRATCH --adapter-dir SCRATCH/ad   # smoke
    uv run python scripts/13_gemma_finetune.py                                                       # full
    uv run python scripts/13_gemma_finetune.py --resume                                              # after a crash
"""

from __future__ import annotations

import argparse
import importlib.util
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
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoProcessor, get_cosine_schedule_with_warmup  # noqa: E402

from benchmark.runmeta import write_run_meta  # noqa: E402
from benchmark.taxonomy import Taxonomy  # noqa: E402


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


qft = _load("qft", "10_qwen_finetune.py")   # target_json, resize, PROMPT_FILE
gz = _load("gz", "12_gemma_zeroshot.py")    # load_model (bf16 on GPU, per-layer embeddings on CPU)

MODEL_ID = "google/gemma-4-E4B-it"
KEEP = 96   # positions whose logits are computed; the answer (JSON + end-of-turn) is ~45 tokens at the sequence end


def lm_loss(model, batch):
    """Cross-entropy on the answer tokens only. Gemma's 262k vocabulary makes full-sequence logits multi-GB, so
    with LEFT padding (answer at the end) logits are computed for just the last KEEP positions."""
    labels = batch["labels"]
    assert bool((labels[:, :-KEEP] == -100).all()), "supervised tokens fall outside the kept window"
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**inputs, logits_to_keep=KEEP).logits[:, :-1].float()
    tgt = labels[:, -KEEP:][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)


class Collator:
    def __init__(self, proc, prompt, targets, ds):
        self.proc, self.prompt, self.targets, self.ds = proc, prompt, targets, ds
        u = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
        gen = proc.apply_chat_template(u, tokenize=False, add_generation_prompt=True)
        no = proc.apply_chat_template(u, tokenize=False, add_generation_prompt=False)
        self.suffix = gen[len(no):]                      # "<|turn>model\n"
        assert self.suffix and gen.startswith(no)
        self.marker = proc.tokenizer.encode(self.suffix, add_special_tokens=False)

    def text(self, cls):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": self.targets[cls]}]}]
        t = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        gen = self.proc.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        assert t.startswith(gen), "training text must start with the exact inference prompt"
        return t

    def __call__(self, rows):
        ims = [qft.resize(self.ds[int(r.idx)]["image"].convert("RGB")) for r in rows]
        texts = [self.text(r.clean_label) for r in rows]
        enc = self.proc(text=texts, images=[[im] for im in ims], return_tensors="pt", padding=True)
        labels = torch.full_like(enc["input_ids"], -100)
        m = len(self.marker)
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = max(j for j in range(len(ids) - m + 1) if ids[j:j + m] == self.marker) + m
            labels[i, start:] = enc["input_ids"][i, start:]
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def selfcheck(model, coll, rows):
    model.eval()
    batch = coll(rows).to("cuda")
    with torch.inference_mode():
        a = model(**batch, use_cache=True).loss.item()
        b = model(**batch, use_cache=False).loss.item()
        c = lm_loss(model, batch).item()
    print(f"selfcheck loss use_cache=True: {a:.5f}  use_cache=False: {b:.5f}  |diff|={abs(a - b):.2e}")
    print(f"selfcheck windowed-logits loss: {c:.5f}  vs full-sequence HF loss {a:.5f}  |diff|={abs(a - c):.2e}")
    lab = batch["labels"][0]
    print("supervised tokens in first row decode to:", repr(coll.proc.tokenizer.decode(lab[lab != -100])))
    print("sequence length (incl. image tokens):", batch["input_ids"].shape[1])
    return abs(a - b) < 1e-3 and abs(a - c) < 2e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", default=str(ROOT / "results" / "gemma4_e4b_lora"))
    ap.add_argument("--adapter-dir", default=str(ROOT / "data" / "models" / "gemma4_e4b_lora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=40)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6, help="validations without improvement before stopping (0 = off)")
    ap.add_argument("--min-delta", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    out, adir = Path(args.out), Path(args.adapter_dir)
    ck = adir / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    tr = sp[sp.split == "train"].reset_index(drop=True)
    va = sp[sp.split == "val"].sample(frac=1, random_state=args.seed).head(args.val_limit)
    assert not (set(tr.idx) & set(sp[sp.split == "test"].idx)), "train/test overlap"
    tax = Taxonomy.load()
    targets = {c: qft.target_json(tax, c) for c in sorted(tr.clean_label.unique())}
    prompt = qft.PROMPT_FILE.read_text().strip()

    ds = load_dataset("inuwamobarak/african-atire", split="train")
    proc = AutoProcessor.from_pretrained(args.model_id)
    proc.tokenizer.padding_side = "left"   # answer must sit at the end of the sequence (see lm_loss)
    proc.image_processor.image_seq_length = args.image_tokens
    if hasattr(proc, "image_seq_length"):
        proc.image_seq_length = args.image_tokens
    model = gz.load_model(args.model_id)
    coll = Collator(proc, prompt, targets, ds)

    if args.selfcheck:
        ok = selfcheck(model, coll, list(tr.head(4).itertuples()))
        print("SELFCHECK", "PASS" if ok else "FAIL: loss depends on use_cache")
        return

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(tr) / (args.batch * args.accum))
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)

    step, log, best, bad = 0, [], float("inf"), 0
    if args.resume and (ck / "state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(ck / "adapter" / "adapter_model.safetensors")))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, log, best, bad = st["step"], st["log"], st["best"], st["bad"]
        print(f"resumed from step {step} (best val {best:.5f}, patience used {bad})")

    write_run_meta(out, kind="finetune_vlm_lora", model=args.model_id, prompts={"structured": qft.PROMPT_FILE},
                   decoding={"dtype": "bfloat16", "max_image_side_preresize": qft.MAX_SIDE,
                             "image_token_budget": args.image_tokens},
                   params={"epochs": args.epochs, "max_steps": args.max_steps, "total_steps_planned": total,
                           "batch": args.batch, "accum": args.accum, "lr": args.lr, "lora_rank": args.rank,
                           "n_train": len(tr), "seed": args.seed, "targets": targets, "adapter_dir": str(adir),
                           "early_stopping": {"patience": args.patience, "min_delta": args.min_delta,
                                              "val_every": args.val_every, "val_items": len(va)},
                           "per_layer_embeddings_on_cpu": True},
                   notes="Targets use the FIRST listed country/people/technique per class from the draft taxonomy; "
                         "confidence fixed at 1.0. Same recipe as the Qwen fine-tune.")

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for b in range(0, len(va), args.batch):
                batch = coll(list(va.iloc[b:b + args.batch].itertuples())).to("cuda")
                tot += lm_loss(model, batch).item() * len(batch["input_ids"])
                n += len(batch["input_ids"])
        model.train()
        return tot / n

    def save_ckpt():
        (ck / "adapter").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ck / "adapter")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "log": log,
                    "best": best, "bad": bad}, ck / "state.pt.tmp")
        os.replace(ck / "state.pt.tmp", ck / "state.pt")   # atomic: a crash mid-save keeps the previous one

    t0, stopped = time.time(), "completed schedule"
    model.train()
    while step < total and not stopped.startswith("early"):
        epoch = step // steps_per_epoch
        order = list(tr.sample(frac=1, random_state=args.seed + epoch).itertuples())
        for mb in range((step % steps_per_epoch) * args.batch * args.accum, len(order), args.batch * args.accum):
            window = order[mb:mb + args.batch * args.accum]
            loss_acc = 0.0
            for b in range(0, len(window), args.batch):
                batch = coll(window[b:b + args.batch]).to("cuda")
                loss = lm_loss(model, batch) * (len(batch["input_ids"]) / len(window))
                loss.backward()
                loss_acc += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": loss_acc,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
            if step % args.val_every == 0 or step == total:
                v = val_loss()
                rec["val_loss"] = v
                if v < best - args.min_delta:
                    best, bad = v, 0
                    (adir / "best").mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(adir / "best")
                else:
                    bad += 1
                if args.patience and bad >= args.patience:
                    stopped = f"early stop at step {step}: val loss flat for {bad} validations (best {best:.5f})"
            log.append(rec)
            print(rec, flush=True)
            if step % args.ckpt_every == 0:
                save_ckpt()
            if step >= total or stopped.startswith("early"):
                break
    adir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adir / "final")
    save_ckpt()
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_summary.json").write_text(json.dumps({"stopped": stopped, "steps_done": step, "planned": total,
                                                         "best_val_loss": best}, indent=1))
    print(f"{stopped}. adapters saved to {adir}/final and {adir}/best")


if __name__ == "__main__":
    main()
