# The Grammar of Luxury

*A multilingual model of luxury sentiment in hospitality reviews.*

SoftUni Deep Learning final project — Avgustina Daskalova

---

## What this is

An aspect-sentiment classifier for luxury-hospitality review text, working across English, Spanish, Portuguese, and Italian.

Its analytic core is an authored taxonomy of 74 terms across five aspects — service, culinary precision, ambience, ingredients, experience. The taxonomy is domain knowledge, not learned. It is written in English, and the corpus is not; whether it survives the crossing into three further languages is the first question the notebook asks, and §2 answers it with a native-speaker validation rather than an assertion.

The taxonomy then becomes the labelling instrument. Through distant supervision, the terms that survive the crossing test generate aspect labels at scale — no hand annotation — and sentiment comes from the ratings the guests left themselves.

The model is built in two stages. A frozen LaBSE encoder with two small trained heads establishes the baseline; the encoder is then fine-tuned end-to-end on both tasks, and the comparison between the two stages is the project's central result.

---

## Results

| | frozen encoder | fine-tuned encoder | gain |
|---|---|---|---|
| aspect macro-F1 (test) | 0.611 | **0.808** | +0.197 |
| sentiment macro-F1 (test) | 0.851 | **0.899** | +0.048 |

Baselines, both as macro-F1: 0.429 for aspect (predict every aspect for every review) and roughly 0.43 for sentiment (always predict positive). The sentiment baseline is worth stating carefully — the majority class is 75% of reviews, so always-positive reaches 75% *accuracy* while never identifying a single negative; macro-averaging is what exposes that. The threshold is fixed at 0.5 throughout and never tuned on the test set.

The two gains are the finding, and their asymmetry is the point. Within the aspect task, fine-tuning lifts what the frozen model handled worst — ingredients +0.314, experience +0.262 — and what it handled best least, service +0.067. Across tasks, the same pattern holds one level up: sentiment, which the frozen encoder already read well, gained four times less than aspect. Adaptation does not add generic capacity, or everything would gain alike. It repairs specific semantic distinctions, and where few are broken there is little to repair.

The per-language sentiment result was not predicted: the gain falls entirely on English (+0.103) and Spanish (+0.065) and reverses on Italian (−0.050) and Portuguese (−0.046), tracking the number of negative reviews each language brings to the test set — 30, 17, 12, and 3 respectively. Fine-tuning helped where there was minority-class data to learn from.

These figures are agreement with a distantly-supervised answer key, not accuracy in an absolute sense. §5.4 audits the key itself against eighty reviews annotated by hand: it recovers 93 of the 165 aspect-positives a reader finds, and scores macro-F1 0.421 against that human judgement. The frozen-versus-fine-tuned comparisons survive — both sides were scored against the same key — but no absolute number should be read as the model's competence.

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
| `finetune_labse_grammar_of_luxury.ipynb` | The §7 aspect fine-tune, run on Colab. |
| `finetune_labse_sentiment.ipynb` | The same procedure repeated for sentiment — §7's second condition. |

Both fine-tune notebooks are committed with their outputs visible, so their results can be read without a GPU. The fine-tuned checkpoints (471M parameters each) are not committed.

---

## Reproducibility

- Run the notebook top to bottom from a fresh kernel (Kernel → Restart Kernel and Run All Cells). Execution counts should read sequentially; if they don't, the notebook has not been run clean.
- The 75 expert judgements are recorded verbatim in the notebook as a literal dictionary, so the threshold calibration reproduces without re-consulting a human reader. The 80 hand annotations of §5.4 are recorded the same way.
- The similarity threshold is defined once, in §2.3, and inherited downstream.
- `SEED = 42` throughout: the train/validation/test split, the head initialisations, and the error sampling in §5.3 all reproduce exactly.
- Both fine-tunes use the identical split, exported to CSV from this notebook and uploaded to Colab, so all three models are compared on the same rows.

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

**Colab (GPU) — the §7 fine-tunes**

Both were run on Google Colab's default runtime with an NVIDIA T4. Library versions there are managed by Google and are not pinned by this project; the sentiment notebook records the versions it ran under (torch 2.11.0+cu128, transformers 5.13.1) in its first cell. Unfreezing all 471M parameters of LaBSE requires a GPU, and nothing else in the project does.

---

*Code structure, debugging, and editorial review were assisted by Anthropic's Claude. All research design, data decisions, linguistic judgements, native-speaker annotation, and written analysis are the author's own.*
