# The Grammar of Luxury

*A multilingual model of luxury sentiment in hospitality reviews.*

SoftUni Deep Learning final project — Avgustina Daskalova

---

## What this is

An aspect-sentiment classifier for fine-hospitality review text, working across
English, Spanish, Portuguese, and Italian. The analytic core is an authored
taxonomy of 74 terms across five aspects — service, culinary precision,
ambience, ingredients, experience — carried into three further languages by
multilingual sentence embeddings rather than by translation.

The taxonomy is domain knowledge, not learned. Whether it survives the crossing
is the first question the notebook asks, and §2 answers it with a
native-speaker validation rather than an assertion.

---

## Data

| Source | Languages | Reviews used |
|---|---|---|
| [MAiDE-up](https://github.com/MihaelaGaman/MAiDE-up) (Ignat et al., 2024) | en, es, it | 1,000 each |
| [Brazilian Portuguese Hotel Reviews Corpus](https://data.mendeley.com/datasets/2w3kvrg97m/1) (Souza, Oliveira & Moreira, 2018) | pt | 1,000 |

4,000 reviews in total, balanced by language. Only real (human-written) reviews
are kept from MAiDE-up; its AI-generated half is discarded. Ratings from both
sources are normalised to a common 0–1 scale.

### ⚠ One manual download is required

MAiDE-up downloads automatically. The Brazilian corpus does not — Mendeley
serves it behind a browser download button:

1. Visit https://data.mendeley.com/datasets/2w3kvrg97m/1
2. Download `tripadvisorBrazilianHotelREviews.zip` (98.8 MB, CC BY 4.0)
3. Place it in `data/`

The notebook unzips and loads it from there. Every other dataset is fetched by
the notebook itself.

---

## Method, in brief

1. **Taxonomy** — 74 English terms across five aspects, authored, not learned.
2. **Crossing** — LaBSE embeds both the taxonomy and a per-language candidate
   vocabulary drawn from the corpus; each term is matched to its nearest word
   in Spanish, Portuguese, and Italian.
3. **Validation** — 75 proposals are judged by a native-level reader of all
   three target languages against a usage question: *would a guest writing in
   this language use this word?* Not dictionary equivalence.
4. **Calibration** — the acceptance threshold is chosen by testing every cut
   against those 75 verdicts. 0.90 reaches 96.0% agreement; the best cut
   anywhere on the scale is 0.890, at the same 96.0%.
5. **Filtered taxonomy** — 34 of 74 terms (46%) cross into at least one
   language, overwhelmingly single words, weighted toward ambience and service.

---

## Secondary component: the narrating voice

The classifier produces sentiment reports. A companion text-to-speech model
narrates them in the author's own voice — a fine-tuned TTS model trained on
her recorded speech.

The pairing is not decorative. The project's subject is a professional register
that exists mainly as spoken language: the tour director's account of a place,
delivered aloud. A report about how luxury announces itself in four languages,
read in the voice of someone who has spent twenty-two years announcing it, is
the natural output form rather than a novelty.

This component is documented separately as it develops; the classifier in this
notebook stands on its own.

---

## Reproducibility

- Run the notebook top to bottom from a fresh kernel
  (Kernel → Restart Kernel and Run All Cells).
- The 75 expert judgements are recorded verbatim in the notebook as a literal
  dictionary, so the calibration is reproducible without re-consulting a human.
- Random sampling uses a fixed seed; the same 25 terms are drawn each run.
- The similarity threshold is defined once, in §2.3, and inherited downstream.

---

## Environment

| | |
|---|---|
| Python | 3.13.9 |
| OS | Windows 11 (AMD64) |
| CPU cores | 4 |
| GPU | none — CPU-only (`torch 2.11.0+cpu`, CUDA unavailable) |
| torch | 2.11.0 |
| transformers | 5.5.4 |
| sentence-transformers | 5.4.1 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |

Encoder: `sentence-transformers/LaBSE`, downloaded on first run
(~1.8 GB, cached thereafter).

### A note on hardware

All work in §§1–2 runs comfortably on CPU; embedding 4,000 reviews and a
per-language candidate vocabulary takes minutes, not hours. Model training in
later sections is designed around the same constraint — the encoder is used as
a frozen feature extractor and only a light classification head is trained, so
the notebook remains runnable end to end on a machine without a GPU.

---

## Repository

```
The-Grammar-of-Luxury/
├── the-grammar-of-luxury.ipynb    the project
├── data/                           corpora (see manual step above)
└── README.md
```

---

## Licence and attribution

Both corpora are used under their original licences and cited in the notebook.
The taxonomy and all analysis are original work.
