# African textile provenance

This project has two aims:
- a benchmark that checks whether AI vision models identify African textiles to their specific provenance (region → people → technique) rather than as a monolithic "Africa";
- models fine-tuned to do this well.

Dataset: [`inuwamobarak/african-atire`](https://huggingface.co/datasets/inuwamobarak/african-atire) (12,363 images, 10 classes, OpenRAIL).

## Setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
uv sync                                           # core + dev (Python 3.12)
uv sync --group vision                            # + DINOv2 classifier deps
```
Hugging Face downloads go to `data/hf_cache/`. The scripts set `HF_HOME` themselves.

## Steps
1. **Expert review pack:** `uv run python scripts/01_fetch_and_sample.py`. This writes:
   - `results/class_counts.csv`;
   - `data/expert_review/expert_review.zip`, containing the Excel workbook, 50 images and the benchmark draft.
2. **Benchmark:** see [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md).
   - Score predictions: `uv run python -m benchmark.scorer --pred P.jsonl --gold G.jsonl --out results/run`
   - Run the scorer's tests: `uv run pytest`
3. **Clean and split:** `scripts/02_clean_data.py` (relabel, dedup), `scripts/03_small_image_sample.py`,
   then `scripts/04_embeddings_and_splits.py` (DINOv2 embeddings, grouped 70/15/15 split).
4. **DINOv2 linear probe:** `uv run python scripts/05_dinov2_probe.py` (needs `uv sync --group vision`).
5. **Report assets:** `uv run python scripts/06_report_assets.py` collects every metric into
   `results/report_assets/summary.json`, writes `tables/*.csv` (the table twin of each figure) and
   draws `figures/*.png`. Re-run after any upstream step.
6. **Qwen3-VL-4B:** `scripts/07_qwen_zeroshot.py` (zero-shot; `--adapter DIR` evaluates a fine-tuned adapter) and
   `scripts/10_qwen_finetune.py` (LoRA on the train split; needs `uv sync --group vision --group vlm`).
7. **Model report assets:** `uv run python scripts/11_model_report_assets.py` writes the comparison figures (fig7-fig9),
   per-model figures (fig10-fig13b), the example-image figures (fig14-fig15, seeded random draws) and
   `summary_models.json` under `results/report_assets/`.

## Status of the gold labels
`benchmark/taxonomy.yaml` is a **draft**. Idama and Idgo have no gold answers below "African" until the expert defines them. All other entries also await expert verification.
