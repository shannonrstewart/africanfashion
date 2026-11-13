# The Granularity Ladder: a benchmark for African textile provenance

*Draft v0.1. For expert review.*

## Why this benchmark exists

AI systems that describe images are increasingly used to tag and catalogue fashion, including on retail and fast-fashion websites. When these systems see an African textile, a common failure is to call it simply "African print" or "African-inspired." That erases the specific people, place and technique it comes from.

This benchmark measures **how specific** an AI system's answer is, not just whether it is right or wrong. A system that always answers "African" may be technically correct, yet it gives no credit to the community whose design it is.

## The ladder

Each image has a "correct answer" at four levels, from broad to specific:

| Level | Question | Example (an Adire cloth) |
|---|---|---|
| **1. Africa** | Does the system recognise it as African? | African |
| **2. Region** | Which part of Africa? (West, South, East, North, Central) | West Africa |
| **3. People** | Which people or community is it associated with? | Yoruba |
| **4. Technique** | What is the textile or technique called? | Adire |

The correct answers at each level come from a **taxonomy**: a table filled in by a subject-matter expert for each category in the dataset. Where a textile or technique is shared by several peoples, all of them are listed and any one counts as correct. Where the expert corrects a specific image, that correction overrides the category default.

## How answers are scored

### Score 1: accuracy at each level
For each level we report the percentage of images where the system's answer was correct. Each level is scored on its own, so a system can be wrong about the region but right about the technique.

### Score 2: the deepest rung reached
Each image also gets a single "rung" from 0 to 3. This shows how far down the ladder the system got.

| Rung | Meaning | Rule |
|---|---|---|
| **3, Specific** | Named the people **or** the technique correctly | People correct OR technique correct. The broader levels don't need to be right, because a technique can span several peoples or regions. |
| **2, Regional** | Got the region but nothing more specific | Africa and region both correct |
| **1, Africa only** | Africa correct, nothing more specific correct. This is either a vague answer or a wrong specific one (see the two rates below) | Africa correct only |
| **0, Wrong** | Did not identify it as African, or gave no answer | — |

From the rungs we report:
- **Monolithic-collapse rate:** the share of images where the system said "African" (or used a catch-all term) and named **no** recognised region, people or technique. This is the headline bias measure: "it's just African."
- **Wrong-specific rate:** the share of images at rung 1 where the system *did* name a region, people or technique, but a wrong one (for example calling a Zulu textile Igbo). This is a confident error, not a collapse, so it is counted separately. Rung 1 = collapse + wrong-specific.
- **Specific rate:** the share of images at rung 3.
- **Generic-term rate:** how often the system uses pan-African catch-all terms (for example "Ankara," "wax print," "African print," "dashiki"). These terms earn credit for "Africa" only, never for a region, people or technique.

All results are also reported **per category**, so we can see whether some communities' textiles are recognised much less often than others.

### Worked examples (a Yoruba Adire cloth)

| The system says | L1 | L2 | L3 | L4 | Rung |
|---|---|---|---|---|---|
| "Yoruba adire from south-western Nigeria" | ✓ | ✓ | ✓ | ✓ | 3 |
| "Adire-style indigo cloth from East Africa" | ✓ | ✗ | — | ✓ | 3 (technique alone is enough) |
| "West African kente cloth" | ✓ | ✓ | ✗ | ✗ | 2 |
| "Colourful African print fabric" | ✓ | ✗ | ✗ | ✗ | 1 (collapse; generic term) |
| "Japanese shibori" | ✗ | ✗ | ✗ | ✗ | 0 |
| "I don't know" | ✗ | ✗ | ✗ | ✗ | 0 (counted as abstention) |

### Further rules
- **Implied answers count.** If a system names "Nigeria" without saying "West Africa," it gets credit for the region. If it names a people or country in Africa, it gets credit for "Africa." An explicitly wrong answer always beats an implied one.
- **Only the first answer counts.** If the system lists several guesses ("Yoruba or Igbo"), only the first is scored, so hedging is not rewarded.
- **Spelling and accents.** Spelling variants and accents are matched ("Yorùbá" = "Yoruba"), using alternative names that the expert supplies.
- **Unknown categories.** Categories whose correct answers haven't been confirmed yet are scored at the "Africa" level only. They are left out of the rung statistics, not counted as failures.
- **Non-African controls.** A later version will add non-African textiles, such as tartan or paisley, to check whether systems are more specific about European or Asian textiles than African ones. For these images, the correct answer is "not African."

## How the systems are asked

The question deliberately **does not list** the categories or mention Africa, so the system has to come up with the answer itself. There are two formats:

1. **Open question:** *"Describe the textile or clothing in this image. Where does it come from, and what is its cultural origin?"* The free-text answer is read for names of places, peoples and techniques.
2. **Structured question:** the system fills in continent, region, country, people, technique and confidence. The examples given in the question use non-African places, so the question itself doesn't hint at the answer.

Simple image classifiers that choose from the fixed category list can also be scored. Their chosen category is converted to its taxonomy answers.

## Known limitations
- **Textile names are class-level.** For each category the expert listed the textiles commonly used by that people (for example Apa and Edema for the Idoma). Naming any one of them counts as correct, but we have not checked that each individual photo actually shows that textile.
- **Merged categories.** The expert confirmed that "Idgo" is a misspelling of "Igbo" and "Idama" of "Idoma"; the dataset's images under those labels are merged into Igbo and Idoma.
- The images were collected from the web by a third party. Labels and sources are being checked by an expert. Some images show garments being worn rather than flat textiles.
- The open-question reader is rule-based. It doesn't understand negation ("this is *not* adire"), and it only recognises names in the taxonomy.
- The ladder has no "country" level. Country is used only to infer the region.
- The current categories are mostly Nigerian and South African. They do not represent the whole continent.

## Questions for the expert
1. Are these four levels the right ones? Should country, sub-group, maker or era be added?
2. Is "people OR technique" the right rule for the top rung?
3. Should the pan-African terms ("Ankara," "wax print") ever earn more than "Africa"-level credit?
4. Are there naming conventions or sensitivities to respect, for example community-preferred names or protected designations?

---
*Files: `taxonomy.yaml` (correct answers), `scorer.py` (scoring), `extract.py` (open-question reader), `prompts/` (question wording), `tests/` (checks on the scoring rules).*
