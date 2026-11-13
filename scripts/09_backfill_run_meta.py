"""One-off: write reconstructed run_meta.json for runs made before benchmark/runmeta.py existed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.runmeta import write_run_meta  # noqa: E402

P = ROOT / "benchmark" / "prompts"
Q = "Qwen/Qwen3-VL-4B-Instruct"
dec = {"do_sample": False, "max_new_tokens": {"structured": 200, "open": 300}, "dtype": "bfloat16",
       "max_image_side": 768, "batch": 8}
write_run_meta(ROOT / "results/qwen3vl_4b_zeroshot", kind="zeroshot_vlm", model=Q, reconstructed=True,
               prompts={"structured": P / "structured.txt", "open": P / "open_ended.txt"}, decoding=dec,
               params={"n_items": 1260}, notes="Full test split. Run date 2026-11-13.")
write_run_meta(ROOT / "results/qwen3vl_4b_smoke", kind="zeroshot_vlm", model=Q, reconstructed=True,
               prompts={"structured": P / "structured.txt", "open": P / "open_ended.txt"}, decoding=dec,
               params={"n_items": 16}, notes="Smoke test, 2 images per class.")
write_run_meta(ROOT / "results/diag_null_forced", kind="diagnostic_vlm", model=Q, reconstructed=True,
               prompts={"custom": ROOT / "results/diag_null_forced/prompt.txt"},
               decoding={**dec, "max_new_tokens": {"custom": 200}},
               params={"n_items": 237, "ids_file": "results/diag_null_forced/ids.txt"},
               notes="Items that returned all-null in the structured run, re-asked with a forced-guess "
                     "variant of structured.txt. Diagnostic only; not a benchmark protocol.")
write_run_meta(ROOT / "results/dinov2_probe", kind="linear_probe", model="facebook/dinov2-base",
               reconstructed=True, params={"features/C": "see metrics.json", "n_test": 1260},
               notes="Frozen DINOv2 + logistic regression, chosen on validation.")
print("ok")
