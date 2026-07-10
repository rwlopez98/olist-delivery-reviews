# Olist: Does Late Delivery Drive Low Review Scores — and Why?

A single-notebook analysis of the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) built as a data/AI showcase. It answers one question end-to-end: **late delivery → low review score, and an LLM explains the mechanism.**

**Arc:** SQL proves late→bad (quantitative) → two sentiment methods quantify it → an LLM themes the negative reviews (generative).

## Headline findings

- **On-time orders average 4.30★; late orders average 2.57★** — a 1.73-star collapse.
- On-time reviews are 62% 5★; late reviews are **46% 1★**.
- Negative reviews are **overwhelmingly about the parcel, not the product**: `never_arrived` + `late_delivery` dominate the LLM theme ranking, and those themes are ~90% "late" on the independent delivery-data split — the generative and quantitative layers agree without being told to.
- Sentiment external validity vs. star ground truth: pt-BR transformer **75.0%**, LeIA (Portuguese VADER) **64.3%**.
- 96.9% of customers order once — so there is **no retention story** (surfaced to close that door, not open it).

## How it's built

| Layer | Tool | What it owns |
|---|---|---|
| bronze→silver→gold | **SQL (DuckDB → Zerve)** | joins, grain resolution, exclusions, order-grain aggregation |
| features, sentiment, theming, charts | **pandas / transformers / Anthropic / Seaborn+Plotly** | the analytical + generative layers |

- **Grain:** Gold = one row per review (`review_id`), delivery facts joined on. See [ADR-0002](docs/adr/0002-review-dedup-among-delivered-orders.md).
- **GenAI theming:** Anthropic SDK in a batched loop, structured output re-joins the dataframe. See [ADR-0001](docs/adr/0001-llm-theming-via-anthropic-sdk.md).
- **Reproducible:** sentiment + LLM outputs cached to `data/cache/*.pkl` keyed to `review_id`; a plain run loads cache and hits no API.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# put the 9 Kaggle CSVs in data/raw/  (kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
# put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...
python smoke_test.py        # verify the 4 environment walls
python dev_notebook.py      # runs top-to-bottom from cache (set PYTHONUTF8=1 on Windows)
```

Flip `REGENERATE_SENTIMENT` / `REGENERATE_THEMES` in `dev_notebook.py` to re-run the GenAI steps instead of loading cache.

## Layout

```
sql/            bronze→silver→gold (standard dialect; ${DATA_PREFIX} is the only source binding)
src/olist.py    all logic — each function ports into a Zerve block unchanged
dev_notebook.py offline notebook / Zerve port template
docs/adr/       architecture decision records
CONTEXT.md      glossary   ·   PLAN.md   build spec
```

See [PLAN.md](PLAN.md) for the full build spec and the offline→Zerve porting plan.
