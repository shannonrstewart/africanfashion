"""Tests for the granularity-ladder scorer and the free-text extractor.

These use a small fixture taxonomy so the tests don't depend on the draft
gold values (which the expert will change).
"""

from pathlib import Path

import pytest

from benchmark.extract import extract_from_text
from benchmark.scorer import normalize_prediction, score, score_item, gold_for
from benchmark.taxonomy import Taxonomy

FIXTURE = """
version: test
status: test
regions:
  West: [west africa, west african]
  South: [southern africa]
  East: [east africa]
countries:
  Nigeria: {region: West, aliases: [nigeria, nigerian]}
  Ghana: {region: West, aliases: [ghana, ghanaian]}
  South Africa: {region: South, aliases: [south africa, south african]}
peoples:
  Yoruba: {region: West, aliases: [yoruba]}
  Tiv: {region: West, aliases: [tiv]}
  Akan: {region: West, aliases: [akan, ashanti]}
  Xhosa: {region: South, aliases: [xhosa]}
  Sotho: {region: South, aliases: [sotho, basotho]}
  Maasai: {region: East, aliases: [maasai]}
techniques:
  Adire: {peoples: [Yoruba], region: West, aliases: [adire, adire eleko]}
  Anger: {peoples: [Tiv], region: West, aliases: ["a'nger", anger cloth]}
  Kente: {peoples: [Akan], region: West, aliases: [kente]}
  Shweshwe: {peoples: [Sotho, Xhosa], region: South, aliases: [shweshwe, seshoeshoe]}
generic_terms: [ankara, african print, wax print]
classes:
  Adire: {region: West, peoples: [Yoruba], techniques: [Adire]}
  Tiv: {region: West, peoples: [Tiv], techniques: [Anger]}
  Shweshwe: {region: South, peoples: [Sotho, Xhosa], techniques: [Shweshwe]}
  Mystery: {region: null, peoples: [], techniques: []}
"""


@pytest.fixture(scope="module")
def tax(tmp_path_factory) -> Taxonomy:
    p = tmp_path_factory.mktemp("tax") / "taxonomy.yaml"
    p.write_text(FIXTURE)
    return Taxonomy.load(p)


def run(pred, cls, tax, **kw):
    return score_item(normalize_prediction(pred, tax, **kw), gold_for({"id": 1, "class": cls}, tax))


# --- rung rules ---------------------------------------------------------------
def test_full_correct_is_rung3(tax):
    s = run({"continent": "Africa", "region": "West Africa", "people": "Yoruba", "technique": "Adire"}, "Adire", tax)
    assert (s["l1_africa"], s["l2_region"], s["l3_people"], s["l4_technique"], s["rung"]) == (True, True, True, True, 3)


def test_monolithic_answer_is_rung1(tax):
    s = run({"continent": "Africa", "region": None, "people": None, "technique": None}, "Adire", tax)
    assert s["rung"] == 1 and s["l2_region"] is False


def test_region_only_is_rung2(tax):
    s = run({"continent": "Africa", "region": "West Africa", "people": "Akan", "technique": "Kente"}, "Adire", tax)
    assert s["rung"] == 2 and s["l3_people"] is False and s["l4_technique"] is False


def test_technique_correct_people_wrong_is_rung3(tax):
    """Either-or rule: technique alone reaches the top rung."""
    s = run({"continent": "Africa", "region": "Southern Africa", "people": "Zulu", "technique": "Shweshwe"}, "Shweshwe", tax)
    assert s["l3_people"] is False and s["l4_technique"] is True and s["rung"] == 3


def test_wrong_region_correct_technique_still_rung3(tax):
    """Upper rungs need not be correct for rung 3."""
    s = run({"continent": "Africa", "region": "East Africa", "people": None, "technique": "Adire"}, "Adire", tax)
    assert s["l2_region"] is False and s["rung"] == 3


def test_people_in_multi_people_gold_set(tax):
    s = run({"continent": "Africa", "region": "Southern Africa", "people": "Xhosa", "technique": None}, "Shweshwe", tax)
    assert s["l3_people"] is True and s["rung"] == 3


def test_wrong_continent_is_rung0(tax):
    s = run({"continent": "Asia", "region": "South Asia", "people": None, "technique": "batik"}, "Adire", tax)
    assert s["l1_africa"] is False and s["l2_region"] is False and s["rung"] == 0


def test_other_continent_region_not_mistaken_for_african_region(tax):
    p = normalize_prediction({"continent": "Asia", "region": "West Asia"}, tax)
    assert p["region"] is None


# --- aliases, normalisation, inference ---------------------------------------
def test_alias_and_accent_matching(tax):
    s = run({"continent": "African", "region": "west african", "people": "Yorùbá people", "technique": "adire eleko"}, "Adire", tax)
    assert s["rung"] == 3 and s["l3_people"] and s["l4_technique"]


def test_apostrophe_insensitive_in_fields(tax):
    s = run({"continent": "Africa", "technique": "A'nger"}, "Tiv", tax)
    assert s["l4_technique"] is True


def test_country_implies_region(tax):
    s = run({"continent": "Africa", "region": None, "country": "Nigeria"}, "Adire", tax)
    assert s["l2_region"] is True and s["rung"] == 2


def test_no_infer_disables_implication(tax):
    s = run({"continent": None, "region": None, "country": "Nigeria"}, "Adire", tax, infer=False)
    assert s["l1_africa"] is False and s["l2_region"] is False


def test_explicit_wrong_region_beats_inference(tax):
    s = run({"continent": "Africa", "region": "East Africa", "country": "Nigeria"}, "Adire", tax)
    assert s["l2_region"] is False


def test_list_answer_scores_first_only(tax):
    s = run({"continent": "Africa", "people": ["Akan", "Yoruba"]}, "Adire", tax)
    assert s["l3_people"] is False


def test_generic_term_is_tracked_but_not_credited(tax):
    p = normalize_prediction({"continent": None, "technique": "Ankara print"}, tax)
    assert p["generic"] is True and p["african"] is True and p["technique"] is None


# --- abstention and unscoreable levels -----------------------------------------
def test_abstain_counted_as_wrong(tax):
    p = normalize_prediction({"continent": "unknown", "region": "N/A", "people": None, "technique": ""}, tax)
    assert all(p["abstain"].values())
    s = score_item(p, gold_for({"id": 1, "class": "Adire"}, tax))
    assert s["rung"] == 0


def test_class_without_gold_is_unscoreable_below_l1(tax):
    s = run({"continent": "Africa", "region": "West Africa", "people": "Yoruba"}, "Mystery", tax)
    assert s["l1_africa"] is True
    assert s["l2_region"] is None and s["l3_people"] is None and s["l4_technique"] is None
    assert s["rung"] is None  # excluded from rung statistics, not counted as collapse


def test_oracle_with_incomplete_gold_classes(tax):
    gold = GOLD + [{"id": 100 + i, "class": "Mystery"} for i in range(4)]
    preds = [{"id": g["id"], "pred_class": g["class"]} for g in gold]
    m, _ = score(preds, gold, tax)
    assert m["overall"]["specific_rate"] == 1.0
    assert m["overall"]["n_rung_scored"] == len(GOLD)


def test_per_item_override(tax):
    g = gold_for({"id": 1, "class": "Adire", "peoples": ["Yoruba", "Akan"]}, tax)
    p = normalize_prediction({"continent": "Africa", "people": "Ashanti"}, tax)
    assert score_item(p, g)["l3_people"] is True


def test_control_item(tax):
    g = gold_for({"id": 1, "african": False}, tax)
    assert score_item(normalize_prediction({"continent": "Europe"}, tax), g)["l1_africa"] is True
    assert score_item(normalize_prediction({"continent": "Africa"}, tax), g)["l1_africa"] is False


# --- classifier predictions -----------------------------------------------------
def test_pred_class(tax):
    s = run({"pred_class": "Adire"}, "Adire", tax)
    assert s["rung"] == 3
    s = run({"pred_class": "Tiv"}, "Adire", tax)
    assert s["rung"] == 2


# --- aggregate sanity checks ----------------------------------------------------
GOLD = [{"id": i, "class": c} for i, c in enumerate(["Adire", "Tiv", "Shweshwe"] * 4)]


def test_always_african_is_full_collapse(tax):
    preds = [{"id": g["id"], "continent": "Africa"} for g in GOLD]
    m, _ = score(preds, GOLD, tax)
    assert m["overall"]["monolithic_collapse_rate"] == 1.0
    assert m["overall"]["acc_l1_africa"] == 1.0


def test_wrong_specific_is_not_collapse(tax):
    """Rung 1 splits into 'said only Africa' (collapse) and 'named a wrong place/people'."""
    gold = [{"id": 1, "class": "Adire"}, {"id": 2, "class": "Adire"}, {"id": 3, "class": "Adire"}]
    preds = [
        {"id": 1, "continent": "Africa"},                                        # collapse
        {"id": 2, "continent": "Africa", "region": "East Africa", "people": "Maasai"},  # wrong specific
        {"id": 3, "continent": "Africa", "technique": "Ankara print"},           # generic term only: collapse
    ]
    m, items = score(preds, gold, tax)
    assert [i["rung"] for i in items] == [1, 1, 1]
    assert [i["africa_only"] for i in items] == [True, False, True]
    assert m["overall"]["monolithic_collapse_rate"] == pytest.approx(2 / 3)
    assert m["overall"]["wrong_specific_rate"] == pytest.approx(1 / 3)


def test_closed_set_classifier_never_collapses(tax):
    gold = [{"id": 1, "class": "Adire"}, {"id": 2, "class": "Shweshwe"}]
    preds = [{"id": 1, "pred_class": "Shweshwe"}, {"id": 2, "pred_class": "Adire"}]   # both wrong, other region
    m, _ = score(preds, gold, tax)
    assert m["overall"]["monolithic_collapse_rate"] == 0.0
    assert m["overall"]["wrong_specific_rate"] == 1.0


def test_oracle_is_all_rung3(tax):
    preds = [{"id": g["id"], "pred_class": g["class"]} for g in GOLD]
    m, _ = score(preds, GOLD, tax)
    assert m["overall"]["specific_rate"] == 1.0 and m["overall"]["mean_rung"] == 3.0


def test_missing_predictions(tax):
    m, _ = score([], GOLD, tax)
    assert m["n_missing_predictions"] == len(GOLD) and m["overall"]["rung_dist"]["0"] == 1.0


# --- free-text extractor ------------------------------------------------------------
def test_extract_open_ended(tax):
    e = extract_from_text("This is an indigo adire eleko cloth made by Yorùbá women in south-western Nigeria.", tax)
    assert e["people"] == "Yoruba" and e["technique"] == "Adire" and e["country"] == "Nigeria" and e["african"] is True


def test_extract_monolithic(tax):
    e = extract_from_text("A vibrant African print fabric, typical of traditional African fashion.", tax)
    assert e["african"] is True and e["people"] is None and e["technique"] is None and e["generic"] is True


def test_extract_english_anger_not_matched(tax):
    e = extract_from_text("The bold stripes convey anger and power.", tax)
    assert e["technique"] is None
    e = extract_from_text("Tiv a’nger cloth with black and white stripes.", tax)
    assert e["technique"] == "Anger"


def test_extract_bare_region_word_not_matched(tax):
    e = extract_from_text("Worn in the West and the South of the country.", tax)
    assert e["region"] is None and e["african"] is None


def test_expert_gold_and_merged_labels():
    t = Taxonomy.load(Path(__file__).parents[1] / "taxonomy.yaml")
    # Source labels resolve to the merged classes with the expert's answers.
    g = gold_for({"id": 1, "class": "Idgo"}, t)
    assert g["class"] == "Igbo" and g["peoples"] == ["Igbo"] and g["techniques"] == ["Isi agu"]
    g = gold_for({"id": 2, "class": "Idama"}, t)
    assert g["class"] == "Idoma" and g["techniques"] == ["Apa", "Edema"]
    assert t.class_gold("Igala")["techniques"] == ["Achi", "Ebulejonu"]
    assert t.class_gold("Adire")["peoples"] == ["Yoruba"]
    # Any ONE listed textile is correct (set semantics), by name or alias.
    for name, cls in [("Edema", "Idoma"), ("apa", "Idoma"), ("Ebulejonu", "Igala"),
                      ("Isi agu", "Igbo"), ("isiagu", "Igbo"), ("A'nger", "Tiv"), ("Anger", "Tiv")]:
        p = normalize_prediction({"continent": "Africa", "technique": name}, t)
        assert score_item(p, gold_for({"id": 1, "class": cls}, t))["l4_technique"] is True, (name, cls)
    # A textile of another people is not credited.
    p = normalize_prediction({"continent": "Africa", "technique": "Apa"}, t)
    assert score_item(p, gold_for({"id": 1, "class": "Igala"}, t))["l4_technique"] is False
    # Every merged class is now fully scoreable (rung defined).
    for cls in ("Idgo", "Idama", "Igala", "Tiv", "Adire"):
        s = score_item(normalize_prediction({"pred_class": cls}, t), gold_for({"id": 1, "class": cls}, t))
        assert s["rung"] == 3


def test_real_taxonomy_loads():
    t = Taxonomy.load(Path(__file__).parents[1] / "taxonomy.yaml")
    assert len(t.classes) == 8
    assert "Idama" not in t.classes and "Idgo" not in t.classes
    assert t.canonical_class("Idgo") == "Igbo" and t.canonical_class("Idama") == "Idoma"
    assert t.canonical_class("Tiv") == "Tiv"
    for name, c in t.classes.items():
        for p in c.get("peoples") or []:
            assert p in t.people_region, f"{name}: unknown people {p}"
        for k in c.get("techniques") or []:
            assert k in t.techniques, f"{name}: unknown technique {k}"
