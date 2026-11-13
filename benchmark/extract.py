"""Rule-based extractor: map a free-text (open-ended) model answer onto the ladder.

It takes the earliest mention at each level. This is a transparent baseline.
Known limitations:
  - Negation is not handled: "this is not Yoruba adire" is read as Yoruba / Adire.
  - Hedged lists ("Yoruba or Igbo") are credited with the first mention.
  - Only names in taxonomy.yaml are recognised; anything else counts as unrecognised.
An LLM-judge extractor can be swapped in later with the same output schema.
"""

from __future__ import annotations

import re

from .taxonomy import Taxonomy, norm_text

_AFRICA = re.compile(r"(?<![a-z0-9])africa(?:n|ns)?(?![a-z0-9])")


def extract_from_text(text: str, tax: Taxonomy) -> dict:
    t = norm_text(text or "")

    def earliest(level: str) -> str | None:
        hits = tax.tidx[level].find_all(t)
        return min(hits)[1] if hits else None

    region = earliest("region")
    country = earliest("country")
    people = earliest("people")
    technique = earliest("technique")
    generic = bool(tax.tidx["generic"].find_all(t))
    mentions_africa = bool(_AFRICA.search(t))
    african = True if (mentions_africa or region or country or people or technique or generic) else None
    return {"african": african, "region": region, "country": country,
            "people": people, "technique": technique, "generic": generic}
