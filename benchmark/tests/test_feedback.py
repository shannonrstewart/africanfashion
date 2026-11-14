"""Tests for the community-feedback loop (parsing, weighting, voting, image matching)."""

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from benchmark import feedback as fb
from benchmark.imghash import best_match, hash_bytes
from benchmark.taxonomy import Taxonomy

TAX = Taxonomy.load()


def sub(knowledge="", tradition="Tiv — A'nger"):
    m = fb.map_tradition(tradition, TAX)
    return SimpleNamespace(knowledge=knowledge, people=m["people"], technique=m["technique"],
                           class_note=m["note"])


def test_strip_html():
    assert fb.strip_html("<p>A <strong>b</strong>&amp;c</p>") == "A b&c"


def test_pseudonym_is_stable_and_hides_name():
    assert fb.pseudonym("Ada Obi") == fb.pseudonym(" ada obi ")
    assert "ada" not in fb.pseudonym("Ada Obi")


def test_map_tradition():
    assert fb.map_tradition("Yoruba — Adire", TAX)["cls"] == "Adire"
    assert fb.map_tradition("Idoma — Apa / Edema", TAX)["cls"] == "Idoma"
    assert fb.map_tradition("Tiv — A'nger", TAX)["cls"] == "Tiv"
    assert fb.map_tradition("Martian — Zorp", TAX)["cls"] is None


def test_contradiction_detected_but_unrelated_claims_pass():
    bad = sub("A'nger is a traditional Yoruba resist-dyed cloth from Ondo State.")
    assert fb.consistency_check(bad, TAX)["n_contradictions"] == 1
    # documented gap: a false claim naming no taxonomy entity is NOT caught
    gap = sub("Adire is a variation of Indian Kalamkari introduced by Dutch merchants.", "Yoruba — Adire")
    assert fb.consistency_check(gap, TAX)["n_contradictions"] == 0
    good = sub("Handloom strip-weaving using cotton.", "Idoma — Apa / Edema")
    assert fb.consistency_check(good, TAX)["n_contradictions"] == 0


def test_inconsistent_tradition_field_is_contradiction():
    assert fb.consistency_check(sub("", "Yoruba — A'nger"), TAX)["n_contradictions"] == 1


def votes(current, proposals):
    """proposals: list of (submitter, class, weight) for one cluster."""
    return pd.DataFrame([{"submission_id": f"s{i}", "submitter": s, "created_at": str(i), "cluster": 1,
                          "current_label": current, "proposed_class": c, "weight": w}
                         for i, (s, c, w) in enumerate(proposals)])


def decide(current, proposals):
    return fb.aggregate(votes(current, proposals)).iloc[0]


def test_unverified_label_changes_with_enough_votes():
    d = decide("Zulu", [("a", "Xhosa-South Africa", 1), ("b", "Xhosa-South Africa", 1)])
    assert d.decision == "change" and d.new_label == "Xhosa-South Africa"


def test_single_submitter_cannot_change_label():
    assert decide("Zulu", [("a", "Xhosa-South Africa", 5)]).decision == "contested"


def test_verified_label_needs_more_than_expert_weight():
    assert decide("Tiv", [(s, "Igbo", 1) for s in "ab"]).decision == "contested"
    assert decide("Tiv", [(s, "Igbo", 1) for s in "abc"]).decision == "contested"   # 3 vs 3: tie keeps label
    assert decide("Tiv", [(s, "Igbo", 1) for s in "abcd"]).decision == "change"


def test_zero_weight_vote_does_not_tip_the_result():
    p = [(s, "Igbo", 1) for s in "abc"] + [("d", "Igbo", 0)]
    assert decide("Tiv", p).decision == "contested"


def test_one_vote_per_submitter_and_agreement_supports_current():
    p = [("a", "Igbo", 1)] * 5
    assert decide("Zulu", p).decision == "contested"            # same person repeated: one vote
    assert decide("Zulu", [("a", "Zulu", 1)]).decision == "keep"


def test_apply_overrides_relabels_cluster_members():
    manifest = pd.DataFrame({"idx": [1, 2, 3], "cluster": [7, 7, 8], "clean_label": ["Zulu", "Zulu", "Zulu"]})
    split = pd.DataFrame({"idx": [2, 3], "split": ["train", "test"], "clean_label": ["Zulu", "Zulu"]})
    dec = pd.DataFrame([{"cluster": 7, "current_label": "Zulu", "new_label": "Xhosa-South Africa",
                         "decision": "change", "totals": "t", "submissions": "s"}])
    v2, ov = fb.apply_overrides(split, manifest, dec)
    assert list(v2.clean_label) == ["Xhosa-South Africa", "Zulu"]
    assert list(v2.label_source) == ["community_vote", "v1"] and len(ov) == 2


def _img(seed=0):
    rng = np.random.default_rng(seed)
    a = np.kron(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), np.ones((16, 16, 1), dtype=np.uint8))
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return b.getvalue(), a


def test_hash_match_survives_flip_and_reencode():
    data, a = _img(0)
    others = [hash_bytes(_img(s)[0])["hashes"] for s in (1, 2, 3)]
    H = np.array(others + [hash_bytes(data)["hashes"]], dtype=np.uint64)
    b = io.BytesIO()
    Image.fromarray(np.fliplr(a)).save(b, "JPEG", quality=85)
    q = hash_bytes(b.getvalue())
    row, dist = best_match(q["hashes"][0], H)
    assert row == 3 and dist <= fb.MATCH_MAX_DIST
