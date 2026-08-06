# The Grammar of Luxury

*A multilingual model of luxury sentiment in hospitality reviews.*

SoftUni Deep Learning final project — Avgustina Daskalova

---

## What this is

An aspect-sentiment classifier for luxury-hospitality review text, working across English, Spanish, Portuguese, and Italian.

Its analytic core is an authored taxonomy of 74 terms across five aspects — service, culinary precision, ambience, ingredients, experience. The taxonomy is domain knowledge, not learned. It is written in English, and the corpus is not; whether it survives the crossing into three further languages is the first question the notebook asks, and §2 answers it with a native-speaker validation rather than an assertion.

The taxonomy then becomes the labelling instrument. Through distant supervision, the terms that survive the crossing test generate aspect labels at scale — no hand annotation — and sentiment comes from the ratings the guests left themselves.

The model is built in two stages. A frozen LaBSE encoder with two small trained heads establishes the baseline; the encoder is then fine-tuned end-to-end on the same split, and the comparison between the two is the project's central result.

---

## Results

| | frozen encoder | fine-tuned encoder |
|---|---|---|
| aspect macro-F1 (test) | 0.611 | **0.808** |
| sentiment macro-F1 (test) | 0.851 | not fine-tuned |

Baselines: always-positive 0.429 for aspect, majority-class 0.75 for sentiment. The threshold is fixed at 0.5 throughout and never tuned on the test set.

Fine-tuning lifts the aspects the frozen model handled worst — ingredients +0.314, experience +0.262 — and the one it handled best least, service +0.067. That asymmetry is the finding: the vocabulary of *manner* crosses languages in a general multilingual encoder, the vocabulary of *substance* does not, and adapting the encoder is what recovers it.

---

## Data

| Source | Languages | Reviews used |
|---|---|---|
| [MAiDE-up](https://aclanthology.org/2025.findings-naacl.88/) (Ignat, Xu & Mihalcea, 2025) | en, es, it | 1,000 each |
| [Brazilian Portuguese Hotel Reviews Corpus](https://doi.org/10.17632/2w3kvrg97m.1) (Oliveira, Souza, Moreira & Martins, 2021) | pt | 1,000 |

4,000 reviews in total, balanced by language. Only real (human-written) reviews are kept from MAiDE-up; its AI-generated half is discarded. Ratings from both sources are normalised to a common 0–1 scale.

**Corpora are not tracked in this repository.** They are fetched or downloaded locally into an untracked `data/` directory.

### ⚠ One manual download is required

MAiDE-up downloads automatically. The Brazilian corpus does not — Mendeley serves it behind a browser download button:

1. Visit https://data.mendeley.com/datasets/2w3kvrg97m/1
2. Download `tripadvisorBrazilianHotelREviews.zip` (98.8 MB, CC BY 4.0)
3. Place it in a local `data/` directory alongside the notebook

The notebook unzips and loads it from there. Every other dataset is fetched by the notebook itself.

---

## Repository contents

| File | What it is |
|---|---|
| `the-grammar-of-luxury.ipynb` | The project. Runs end to end on CPU. |
| `finetune_labse_grammar_of_luxury.ipynb` | The §7 fine-tune, run on Colab. Committed with its outputs visible, so its results can be read without a GPU. |

The fine-tuned checkpoint (471M parameters) is not committed. The notebook above records the full training run and its results.

---

## Reproducibility

- Run the notebook top to bottom from a fresh kernel (Kernel → Restart Kernel and Run All Cells). Execution counts should read sequentially; if they don't, the notebook has not been run clean.
- The 75 expert judgements are recorded verbatim in the notebook as a literal dictionary, so the threshold calibration reproduces without re-consulting a human reader.
- The similarity threshold is defined once, in §2.3, and inherited downstream.
- `SEED = 42` throughout: the train/validation/test split, the head initialisations, and the error sampling in §5.3 all reproduce exactly.
- The fine-tune uses the identical split, exported to CSV from this notebook and uploaded to Colab, so the frozen and fine-tuned models are compared on the same rows.

---

## Environment

**Local (CPU) — everything except §7**

| | |
|---|---|
| Python | 3.13 |
| torch | 2.11 |
| transformers | 5.5.4 |
| sentence-transformers | 5.4.1 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |

Encoder: `sentence-transformers/LaBSE`, downloaded on first run (~1.8 GB, cached thereafter).

**Colab (GPU) — §7 fine-tune only**

The fine-tune was run on Google Colab's default runtime with an NVIDIA T4. Library versions there are managed by Google and are not pinned by this project; the notebook records the device it ran on in its first cell. Unfreezing all 471M parameters of LaBSE requires a GPU, and nothing else in the project does.

---

*Code structure, debugging, and editorial review were assisted by Anthropic's Claude. All research design, data decisions, linguistic judgements, native-speaker annotation, and written analysis are the author's own.*
