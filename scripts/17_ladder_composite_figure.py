"""Composite figure: per-class ladder heatmaps before and after LoRA fine-tuning.

Left column = zero-shot, right column = fine-tuned; top row = Qwen3-VL-4B (fig10, fig13),
bottom row = Gemma 4 E4B (fig16, fig19). The panels are the existing figures, pasted unchanged
so the numbers cannot drift from the single-model figures.

Writes results/report_assets/figures/fig24_finetuning_ladder_composite.png.
Run (after scripts 06/11):  uv run python scripts/17_ladder_composite_figure.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ra", ROOT / "scripts" / "06_report_assets.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

import matplotlib  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FIG = ra.FIG
PANELS = [["fig10_qwen_zeroshot_ladder.png", "fig13_qwen_lora_ladder.png"],
          ["fig16_gzs_ladder.png", "fig19_gft_ladder.png"]]
LETTERS = [["A", "B"], ["C", "D"]]
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
bold = lambda s: ImageFont.truetype(str(FONTS / "DejaVuSans-Bold.ttf"), s)  # noqa: E731
reg = lambda s: ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), s)  # noqa: E731

imgs = [[Image.open(FIG / f).convert("RGB") for f in row] for row in PANELS]
w, h = imgs[0][0].size
assert all(im.size == (w, h) for row in imgs for im in row), "panels must share one size"
GAP, HEAD, FOOT = 24, 150, 70
W, H = 2 * w + GAP, HEAD + 2 * h + GAP + FOOT
canvas = Image.new("RGB", (W, H), ra.SURF)
d = ImageDraw.Draw(canvas)
TITLE = "Fine-tuning lifts people and textile accuracy from 14% and 5% (Qwen) and 2% and 1% (Gemma) to 91%"
assert d.textlength(TITLE, font=bold(44)) < W - 64, "title too wide"
d.text((32, 34), TITLE, font=bold(44), fill=ra.INK)
d.text((32, 100), "Per-class accuracy at each ladder level on the same 1,260 held-out test images. "
       "Left: zero-shot. Right: LoRA fine-tuned.", font=reg(30), fill=ra.INK2)
for r, row in enumerate(imgs):
    for c, im in enumerate(row):
        x, y = c * (w + GAP), HEAD + r * (h + GAP)
        canvas.paste(im, (x, y))
        d.text((x + w - 70, y + 24), LETTERS[r][c], font=bold(44), fill=ra.INK2)
d.line([(w + GAP // 2, HEAD), (w + GAP // 2, H - FOOT)], fill=ra.GRID, width=2)
d.text((32, H - FOOT + 18), "Panels A-D are fig10, fig13, fig16 and fig19, unchanged.", font=reg(26), fill=ra.MUTED)
out = FIG / "fig24_finetuning_ladder_composite.png"
canvas.save(out)
print("wrote", out, canvas.size)
