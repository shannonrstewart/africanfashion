"""Load taxonomy.yaml and resolve free-form answers to canonical ladder values."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomy.yaml")

ABSTAIN = {"", "unknown", "unsure", "none", "n a", "na", "null", "not sure",
           "cannot determine", "cannot be determined", "uncertain", "not applicable"}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_field(s: str) -> str:
    """Normalise a structured field: lowercase, no accents, no apostrophes, no punctuation."""
    s = _strip_accents(s).lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_text(s: str) -> str:
    """Normalise free text: like norm_field but keeps apostrophes (so "a'nger" != "anger")."""
    s = _strip_accents(s).lower().replace("’", "'")
    return re.sub(r"[^a-z0-9']+", " ", s).strip()


def is_abstain(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple)):
        return all(is_abstain(x) for x in v)
    return norm_field(str(v)) in ABSTAIN


def first_value(v):
    """Predictions may give a list; only the first (primary) answer is scored."""
    if isinstance(v, (list, tuple)):
        for x in v:
            if not is_abstain(x):
                return x
        return None
    return v


class AliasIndex:
    """Maps surface forms to canonical names, supporting exact and whole-word lookup."""

    def __init__(self, normaliser):
        self.norm = normaliser
        self.map: dict[str, str] = {}
        self._pattern: re.Pattern | None = None

    def add(self, alias: str, canonical: str) -> None:
        key = self.norm(alias)
        if key:
            self.map[key] = canonical
        self._pattern = None

    @property
    def pattern(self) -> re.Pattern:
        if self._pattern is None:
            keys = sorted(self.map, key=len, reverse=True)
            alt = "|".join(re.escape(k) for k in keys) or r"(?!x)x"
            self._pattern = re.compile(rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])")
        return self._pattern

    def resolve(self, value: str) -> str | None:
        """Exact match first, then the longest whole-word alias inside the value."""
        key = self.norm(value)
        if key in self.map:
            return self.map[key]
        m = self.pattern.search(key)
        return self.map[m.group(0)] if m else None

    def find_all(self, text_normed: str) -> list[tuple[int, str]]:
        """All (position, canonical) matches in already-normalised text."""
        return [(m.start(), self.map[m.group(0)]) for m in self.pattern.finditer(text_normed)]


@dataclass
class Taxonomy:
    raw: dict
    regions: set[str] = field(default_factory=set)
    country_region: dict[str, str] = field(default_factory=dict)
    people_region: dict[str, str] = field(default_factory=dict)
    techniques: dict[str, dict] = field(default_factory=dict)
    classes: dict[str, dict] = field(default_factory=dict)
    generic_terms: list[str] = field(default_factory=list)
    source_label_map: dict[str, str] = field(default_factory=dict)
    # field (structured) and text (free-text) indices per level
    idx: dict[str, AliasIndex] = field(default_factory=dict)
    tidx: dict[str, AliasIndex] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY) -> "Taxonomy":
        raw = yaml.safe_load(Path(path).read_text())
        t = cls(raw=raw)
        for level in ("region", "country", "people", "technique", "generic"):
            t.idx[level] = AliasIndex(norm_field)
            t.tidx[level] = AliasIndex(norm_text)

        def register(level, canonical, aliases, canonical_in_text=True, canonical_in_field=True):
            if canonical_in_field:
                t.idx[level].add(canonical, canonical)
            if canonical_in_text:
                t.tidx[level].add(canonical, canonical)
            for a in aliases:
                t.idx[level].add(a, canonical)
                t.tidx[level].add(a, canonical)

        for r, aliases in raw["regions"].items():
            t.regions.add(r)
            # A bare "West"/"South" counts only as an exact structured value
            # (see resolve_region). Otherwise "West Asia" would match "West".
            register("region", r, aliases, canonical_in_text=False, canonical_in_field=False)
        for c, spec in raw["countries"].items():
            t.country_region[c] = spec["region"]
            register("country", c, spec.get("aliases", []))
        for p, spec in raw["peoples"].items():
            t.people_region[p] = spec["region"]
            register("people", p, spec.get("aliases", []))
        for k, spec in raw["techniques"].items():
            t.techniques[k] = spec
            # canonical technique names can be common words (e.g. "Anger"),
            # so in free text only explicit aliases count
            register("technique", k, spec.get("aliases", []), canonical_in_text=False)
        t.generic_terms = list(raw.get("generic_terms", []))
        for g in t.generic_terms:
            t.idx["generic"].add(g, g)
            t.tidx["generic"].add(g, g)
        t.classes = raw["classes"]
        t.source_label_map = dict(raw.get("source_label_map") or {})
        return t

    def canonical_class(self, name: str) -> str:
        """Map a source dataset label (e.g. a known misspelling) to its canonical class."""
        return self.source_label_map.get(name, name)

    # --- structured-field resolution -------------------------------------
    def resolve_region(self, value) -> str | None:
        v = first_value(value)
        if is_abstain(v):
            return None
        exact = {norm_field(r): r for r in self.regions}.get(norm_field(str(v)))
        return exact or self.idx["region"].resolve(str(v)) or self.resolve_country_region(v)

    def resolve_country(self, value) -> str | None:
        v = first_value(value)
        return None if is_abstain(v) else self.idx["country"].resolve(str(v))

    def resolve_country_region(self, value) -> str | None:
        c = self.resolve_country(value)
        return self.country_region.get(c) if c else None

    def resolve_people(self, value) -> str | None:
        v = first_value(value)
        return None if is_abstain(v) else self.idx["people"].resolve(str(v))

    def resolve_technique(self, value) -> str | None:
        v = first_value(value)
        return None if is_abstain(v) else self.idx["technique"].resolve(str(v))

    def has_generic(self, *values) -> bool:
        for v in values:
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if x and not is_abstain(x) and self.idx["generic"].pattern.search(norm_field(str(x))):
                    return True
        return False

    def class_gold(self, class_name: str) -> dict:
        c = self.classes[self.canonical_class(class_name)]
        return {
            "african": True,
            "region": c.get("region"),
            "peoples": list(c.get("peoples") or []),
            "techniques": list(c.get("techniques") or []),
        }
