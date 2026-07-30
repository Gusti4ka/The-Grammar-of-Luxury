# The Grammar of Luxury

*A multilingual model of luxury sentiment in hospitality reviews.*

SoftUni Deep Learning final project — Avgustina Daskalova

---

## What this is

An aspect-sentiment classifier for fine-hospitality review text, working across
English, Spanish, Portuguese, and Italian.

Its analytic core is an authored taxonomy of 74 terms across five aspects —
service, culinary precision, ambience, ingredients, experience. The taxonomy is
domain knowledge, not learned. It is written in English, and the corpus is not;
whether it survives the crossing into three further languages is the first
question the notebook asks, and §2 answers it with a native-speaker validation
rather than an assertion.

A companion text-to-speech model, fine-tuned on the author's recorded speech,
narrates the reports the classifier produces. The pairing is not decorative:
the professional register this project studies exists mainly as spoken
language.

---

## Data

| Source | Languages | Reviews used |
|---|---|---|
| [MAiDE-up](https://github.com/MihaelaGaman/MAiDE-up) (Ignat et al., 2024) | en, es, it | 1,000 each |
| [Brazilian Portuguese Hotel Reviews Corpus](https://data.mendeley.com/datasets/2w3kvrg97m/1) (Souza, Oliveira & Moreira, 2018) | pt | 1,000 |

4,000 reviews in total, balanced by language. Only real (human-written) reviews
are kept from MAiDE-up; its AI-generated half is discarded. Ratings from both
sources are normalised to a common 0–1 scale.

**Corpora are not tracked in this repository.** They are fetched or downloaded
locally into an untracked `data/` directory.

### ⚠ One manual download is required

MAiDE-up downloads automatically. The Brazilian corpus does not — Mendeley
serves it behind a browser download button:

1. Visit https://data.mendeley.com/datasets/2w3kvrg97m/1
2. Download `tripadvisorBrazilianHotelREviews.zip` (98.8 MB, CC BY 4.0)
3. Place it in a local `data/` directory alongside the notebook

The notebook unzips and loads it from there. Every other dataset is fetched by
the notebook itself.

---

## Reproducibility

- Run the notebook top to bottom from a fresh kernel
  (Kernel → Restart Kernel and Run All Cells). Execution counts should read
  sequentially; if they don't, the notebook has not been run clean.
- The 75 expert judgements are recorded verbatim in the notebook as a literal
  dictionary, so the threshold calibration reproduces without re-consulting a
  human reader.
- The similarity threshold is defined once, in §2.3, and inherited downstream.
- The encoder is used as a frozen feature extractor; only a light
  classification head is trained.

## Environment

| | |
|---|---|
| Python | 3.13 |
| torch | 2.11 |
| transformers | 5.5.4 |
| sentence-transformers | 5.4.1 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |

Encoder: `sentence-transformers/LaBSE`, downloaded on first run
(~1.8 GB, cached thereafter).
