"""Fetch the african-atire dataset, report class statistics, and build the
expert label-review pack (Excel workbook with embedded thumbnails + image folder + zip).

Usage:
    uv run python scripts/01_fetch_and_sample.py [--seed 42]
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Image as HFImage  # noqa: E402
from datasets import load_dataset  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.drawing.image import Image as XLImage  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

DATASET_ID = "inuwamobarak/african-atire"

# Classes we have open questions about (suspected misspellings + the classes
# they may be confused with) get a larger sample.
SUSPECT_CLASSES = {"Idama", "Idgo", "Idoma", "Igbo"}
N_REGULAR = 3
N_SUSPECT = 8
THUMB_PX = 200

OUT_DIR = ROOT / "data" / "expert_review"
RESULTS_DIR = ROOT / "results"

INSTRUCTIONS = [
    ("Purpose",
     "We are building a benchmark that tests whether AI image models can identify "
     "African textiles to their specific cultural and creative origin (people, "
     "technique) rather than just 'African'. The images come from a public dataset "
     "(Hugging Face: inuwamobarak/african-atire) that was collected from the web. "
     "Before we rely on its labels, we need an expert check of a sample."),
    ("What to do",
     "On the 'Review' sheet, each row is one image. The thumbnail is shown in the "
     "row; full-size copies are in the 'images' folder, named by the same ID "
     "(e.g. AF-001_Adire.jpg). Please fill in the yellow columns. Leave a cell "
     "blank if you do not know; 'Unsure' is a perfectly useful answer."),
    ("Dataset label",
     "The label the dataset gives the image. We have deliberately NOT filled in our "
     "own guesses for people or technique, so as not to influence your answers."),
    ("Label correct?",
     "Y = the dataset label is right; N = it is wrong; Unsure = cannot tell from the image."),
    ("Corrected label",
     "If the label is wrong or imprecise, what should it be? Free text."),
    ("Region / Country",
     "Where this textile or style originates (not where the photo was taken)."),
    ("People / group",
     "The people or community this textile or style is associated with "
     "(e.g. Yoruba, Tiv). More than one is fine; separate with ';'."),
    ("Technique / textile name",
     "The name of the technique or cloth type (e.g. adire eleko, anger cloth, "
     "akwete). More than one is fine; separate with ';'."),
    ("Item type",
     "Is the image mainly a flat textile/fabric, a garment being worn or displayed, "
     "or something else (e.g. beadwork, a collage, a non-textile image)?"),
    ("Confidence",
     "1 = guess, 2 = fairly confident, 3 = certain."),
    ("Categories sheet",
     "One row per dataset category. Please tell us what each category should be "
     "credited to. Our benchmark scores AI answers against these, so your entries "
     "become the 'correct answers'. List several values with ';' where appropriate, "
     "and add alternative spellings or names the AI might use."),
    ("Questions sheet",
     "The 'Questions' sheet lists broader questions about the dataset's categories. "
     "Short answers or pointers are very welcome."),
    ("Benchmark draft",
     "BENCHMARK_draft.md (in the same folder, opens in any text editor) explains how "
     "AI answers will be scored. Comments on the scoring rules are welcome."),
]

CATEGORY_COLUMNS = [
    ("Dataset category", 20, False),
    ("Images in dataset", 10, False),
    ("What does this category refer to?", 30, True),
    ("Region (West/South/East/North/Central)", 16, True),
    ("Country / countries", 20, True),
    ("People(s) / group(s)", 24, True),
    ("Textile / technique name(s)", 28, True),
    ("Alternative names or spellings", 28, True),
    ("Notes", 40, True),
]

QUESTIONS = [
    "The dataset has a category called 'Idama'. What do you think this refers to?",
    "The dataset has a category called 'Idgo'. What do you think this refers to?",
    "Do any two of the dataset's categories (Adire, Idama, Idgo, Idoma, Igala, Igbo, "
    "Tiv, Tswana-Shweshwe, Xhosa-South Africa, Zulu) refer to the same people, "
    "textile or style? If so, which?",
    "Some category names look like peoples and others look like textiles or "
    "techniques. Please fill in the 'Categories' sheet with what each category "
    "should be credited to (region, country, people, textile/technique names).",
    "Do any of the textiles or techniques in these categories span several "
    "peoples or regions? If so, how should we credit them?",
    "How should the provenance of Shweshwe be described?",
    "Some images carry stock-photo watermarks or brand/photographer marks. Does "
    "the source of the images raise any concerns for using them in a research "
    "benchmark?",
    "Our benchmark uses a ladder: Africa > Region (West/South) > People > "
    "Technique. Is anything important missing (e.g. country, sub-group, era, "
    "maker/designer, contemporary vs. traditional)?",
    "From an IP / cultural heritage perspective, are there naming conventions or "
    "sensitivities we should follow when attributing these textiles (e.g. "
    "collective marks, geographical indications, community-preferred names)?",
    "Are there images in this sample that you think should not be in the dataset "
    "at all (wrong category entirely, not African, not a textile, offensive, etc.)?",
    "Anything else we should know about this dataset or these categories?",
]

REVIEW_COLUMNS = [
    # (header, width, filled_by_expert)
    ("ID", 10, False),
    ("Image", 30, False),
    ("Dataset label", 18, False),
    ("Label correct? (Y/N/Unsure)", 14, True),
    ("Corrected label", 20, True),
    ("Region (West/South/East/North/Central/Other)", 16, True),
    ("Country", 16, True),
    ("People / group", 22, True),
    ("Technique / textile name", 24, True),
    ("Item type", 16, True),
    ("Confidence (1-3)", 12, True),
    ("Notes", 40, True),
    ("Dataset row #", 10, False),
]


def class_stats(ds_raw, label_names: list[str]) -> pd.DataFrame:
    """Per-class counts and resolution stats, reading image headers only."""
    rows = []
    for i, ex in enumerate(tqdm(ds_raw, desc="Reading image headers")):
        img = ex["image"]
        try:
            with Image.open(io.BytesIO(img["bytes"])) as im:
                w, h = im.size
                fmt = im.format
        except Exception:  # broken image
            w = h = -1
            fmt = "BROKEN"
        rows.append({"idx": i, "label": label_names[ex["label"]], "w": w, "h": h, "fmt": fmt})
    return pd.DataFrame(rows)


def build_workbook(sample: pd.DataFrame, img_dir: Path, xlsx_path: Path,
                   class_counts: dict[str, int]) -> None:
    wb = Workbook()
    yellow = PatternFill("solid", fgColor="FFF2CC")
    header_fill = PatternFill("solid", fgColor="D9D9D9")
    wrap = Alignment(wrap_text=True, vertical="top")

    # Instructions
    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 100
    ws["A1"] = "Expert label review: African textiles sample"
    ws["A1"].font = Font(bold=True, size=14)
    for r, (k, v) in enumerate(INSTRUCTIONS, start=3):
        ws.cell(r, 1, k).font = Font(bold=True)
        c = ws.cell(r, 2, v)
        c.alignment = wrap
        ws.row_dimensions[r].height = max(30, 15 * (len(v) // 95 + 1))

    # Review
    ws = wb.create_sheet("Review")
    for col, (name, width, expert) in enumerate(REVIEW_COLUMNS, start=1):
        c = ws.cell(1, col, name)
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.fill = yellow if expert else header_fill
        ws.column_dimensions[c.column_letter].width = width
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "C2"

    col_idx = {name: i for i, (name, _, _) in enumerate(REVIEW_COLUMNS, start=1)}
    dv_specs = {
        "Label correct? (Y/N/Unsure)": '"Y,N,Unsure"',
        "Region (West/South/East/North/Central/Other)": '"West,South,East,North,Central,Other,Unsure"',
        "Item type": '"Flat textile,Garment,Beadwork,Other"',
        "Confidence (1-3)": '"1,2,3"',
    }
    n = len(sample)
    for name, formula in dv_specs.items():
        dv = DataValidation(type="list", formula1=formula, allow_blank=True)
        letter = ws.cell(1, col_idx[name]).column_letter
        dv.add(f"{letter}2:{letter}{n + 1}")
        ws.add_data_validation(dv)

    for r, row in enumerate(sample.itertuples(index=False), start=2):
        ws.cell(r, col_idx["ID"], row.id)
        ws.cell(r, col_idx["Dataset label"], row.label)
        ws.cell(r, col_idx["Dataset row #"], int(row.idx))
        for name, _, expert in REVIEW_COLUMNS:
            c = ws.cell(r, col_idx[name])
            c.alignment = wrap
            if expert:
                c.fill = yellow
        # thumbnail
        with Image.open(img_dir / row.filename) as im:
            im.thumbnail((THUMB_PX, THUMB_PX))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        buf.seek(0)
        xl_img = XLImage(buf)
        ws.add_image(xl_img, f"{ws.cell(r, col_idx['Image']).column_letter}{r}")
        ws.row_dimensions[r].height = THUMB_PX * 0.75 + 6  # px -> points

    # Categories (blank, for the expert to define each class's gold answers)
    ws = wb.create_sheet("Categories")
    for col, (name, width, expert) in enumerate(CATEGORY_COLUMNS, start=1):
        c = ws.cell(1, col, name)
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.fill = yellow if expert else header_fill
        ws.column_dimensions[c.column_letter].width = width
    ws.row_dimensions[1].height = 45
    for r, (name, count) in enumerate(class_counts.items(), start=2):
        ws.cell(r, 1, name)
        ws.cell(r, 2, count)
        for col, (_, _, expert) in enumerate(CATEGORY_COLUMNS, start=1):
            c = ws.cell(r, col)
            c.alignment = wrap
            if expert:
                c.fill = yellow
        ws.row_dimensions[r].height = 45

    # Questions
    ws = wb.create_sheet("Questions")
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 80
    ws.column_dimensions["C"].width = 80
    for col, name in enumerate(["#", "Question", "Your answer"], start=1):
        c = ws.cell(1, col, name)
        c.font = Font(bold=True)
        c.fill = header_fill
    for r, q in enumerate(QUESTIONS, start=2):
        ws.cell(r, 1, r - 1)
        ws.cell(r, 2, q).alignment = wrap
        ws.cell(r, 3).fill = yellow
        ws.cell(r, 3).alignment = wrap
        ws.row_dimensions[r].height = max(45, 15 * (len(q) // 75 + 2))

    wb.save(xlsx_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(DATASET_ID, split="train")
    label_names: list[str] = ds.features["label"].names
    print(f"Loaded {len(ds)} rows; classes: {label_names}")

    ds_raw = ds.cast_column("image", HFImage(decode=False))
    stats = class_stats(ds_raw, label_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(RESULTS_DIR / "image_stats.csv", index=False)
    ok = stats[stats.w > 0]
    summary = (
        ok.assign(min_side=ok[["w", "h"]].min(axis=1))
        .groupby("label")
        .agg(n=("idx", "size"), min_side_min=("min_side", "min"),
             min_side_median=("min_side", "median"), min_side_max=("min_side", "max"),
             n_under_128=("min_side", lambda s: int((s < 128).sum())))
        .reindex(label_names)
    )
    summary["n_broken"] = stats[stats.w < 0].groupby("label").size().reindex(label_names).fillna(0).astype(int)
    summary.to_csv(RESULTS_DIR / "class_counts.csv")
    print("\nPer-class statistics (min_side = shorter image edge, px):")
    print(summary.to_string())
    print("\nFormats:", stats.fmt.value_counts().to_dict())

    # --- Stratified sample (excluding broken / tiny images) ---
    rng = np.random.default_rng(args.seed)
    eligible = ok[ok[["w", "h"]].min(axis=1) >= 128]
    picks = []
    for name in label_names:
        pool = eligible[eligible.label == name].idx.to_numpy()
        k = min(N_SUSPECT if name in SUSPECT_CLASSES else N_REGULAR, len(pool))
        picks.extend((name, int(i)) for i in rng.choice(pool, size=k, replace=False))
    # Shuffle so the expert does not review class-by-class (reduces anchoring).
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True)

    rows = []
    for n, (name, idx) in enumerate(picks, start=1):
        sid = f"AF-{n:03d}"
        safe = name.replace(" ", "_")
        fname = f"{sid}_{safe}.jpg"
        im = ds[idx]["image"].convert("RGB")
        im.save(img_dir / fname, quality=92)
        rows.append({"id": sid, "idx": idx, "label": name, "filename": fname,
                     "width": im.width, "height": im.height})
    sample = pd.DataFrame(rows)
    sample.to_csv(OUT_DIR / "manifest.csv", index=False)

    xlsx_path = OUT_DIR / "expert_review.xlsx"
    build_workbook(sample, img_dir, xlsx_path, summary["n"].astype(int).to_dict())

    zip_path = OUT_DIR / "expert_review.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xlsx_path, "expert_review/expert_review.xlsx")
        for f in sorted(img_dir.iterdir()):
            zf.write(f, f"expert_review/images/{f.name}")
        spec = ROOT / "benchmark" / "BENCHMARK.md"
        if spec.exists():
            zf.write(spec, "expert_review/BENCHMARK_draft.md")

    print(f"\nSample: {len(sample)} images")
    print(sample.label.value_counts().reindex(label_names).to_string())
    print(f"\nWrote {xlsx_path}\n      {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
