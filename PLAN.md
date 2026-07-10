# Build Plan — Olist Delivery → Reviews

The section-by-section build spec the handoff referred to but never contained. Architecture is locked by `CLAUDE_CODE_HANDOFF.md`; this file finalizes the *building process*. Terms are defined in [CONTEXT.md](./CONTEXT.md); non-obvious decisions in [docs/adr/](./docs/adr/).

## Phase split (locked)

One direction, no round-trip. Only three artifacts cross offline → Zerve: **SQL text**, the **sentiment pickle** (keyed to `review_id`), and **pandas/viz code**. Now also a **themes pickle** (keyed to `review_id`) — see [ADR-0001](./docs/adr/0001-llm-theming-via-anthropic-sdk.md).

### Phase 1 — Offline (this repo, free compute)
1. **Smoke-test (do first):** LeIA install+run, HF pt-BR model pull, and `anthropic` SDK call with the BYO key. Resolve all three before building on them.
2. **SQL bronze→silver→gold in DuckDB** — standard dialect, carry the *text*. Queries live in `sql/`.
3. **Sentiment A/B** — pt-BR HF transformer + LeIA; infer over the sample; pickle scores keyed to `review_id`.
4. **`get_themes` (Anthropic SDK loop)** — build + test + run here; pickle themes keyed to `review_id`. (Runs offline per ADR-0001, unlike the handoff's original Phase-2 placement.)
5. **Feature engineering + Charts 1–2 (Seaborn)** — pure pandas, portable code.

### Phase 2 — In Zerve (live compute)
1. Paste SQL → runs live against real tables (connector swap only).
2. Load sentiment + themes pickles.
3. Optionally re-run `get_themes` live (`REGENERATE = True`) — identical Anthropic SDK code.
4. Charts 3–4 (Plotly).

## Phase 1 progress
- ✅ Env + venv, deps pinned, smoke-test 3/4 walls (`smoke_test.py`).
- ✅ Data downloaded + verified (9 tables).
- ✅ **SQL bronze→silver→gold built & tested live** (`sql/10_bronze.sql`, `20_silver.sql`, `30_gold.sql`). Gold = **95,645** rows = distinct `review_id`, deterministic across runs ([ADR-0002](./docs/adr/0002-review-dedup-among-delivered-orders.md) fixed a dedup nondeterminism bug). Spine confirmed: on-time avg score **4.30** vs late **2.57** (−1.73★).
- ✅ Feature layer + **Charts 1–2 (Seaborn)** built & visually verified (`src/olist.py`, `reports/figures/`).
- ✅ **Chart 4 (Plotly regional bubble)** built & verified — SP huge/pale, NE/N red outliers.
- ✅ **`get_themes` + theming** built: 400 negatives, stability **95.7%**, delivery themes dominate. **Chart 3** verified (delivery themes carry the "Late" red).
- ✅ Offline notebook assembled (`dev_notebook.py`) — cell-by-cell Zerve port template.
- ✅ Config surface for Zerve embedding: single **`DATA_PREFIX`** source prefix (Bronze SQL tokenized `${DATA_PREFIX}`), and GenAI steps wrapped in explicit **`if REGENERATE_SENTIMENT/THEMES`** load-vs-rebuild blocks (per-task).
- ✅ **Sentiment full run complete** (38,845 reviews): external validity A **75.0%** / B **64.3%**; internal A–B agree **64.4%** (distinct from external — the point); **13,824** disagreement rows surfaced with examples.
- ✅ **Definition of done met:** `dev_notebook.py` runs top-to-bottom from cache in ~8s, no API/inference. All 4 charts, both pickles, both showpieces present.
- ⏭ **Only remaining: port to Zerve** — paste SQL (Bronze block swap / point `DATA_PREFIX` at live tables), load the two pickles, re-render charts. (Offline runs need `PYTHONUTF8=1` on Windows consoles; Zerve is UTF-8 natively.)

## Decisions locked so far
- **Project home:** `OneDrive/Desktop/GH/olist-delivery-reviews` (GitHub-bound, alongside other repos).
- **Data:** Kaggle `olistbr/brazilian-ecommerce` via `kagglehub.dataset_download(...)` (anonymous, no auth) → copied to `data/raw/` (gitignored). All 9 tables verified.
- **LLM path:** Anthropic SDK (`claude-haiku-4-5`), BYO key, Python loop — [ADR-0001](./docs/adr/0001-llm-theming-via-anthropic-sdk.md).
- **Grain:** Gold = one row per Review (locked by handoff).

## Sampling (locked)
- **Charts 1–2:** all ~100k reviews (SQL, no sampling).
- **Sentiment A/B:** full ~40k reviews *with text*, no sampling — one-time cached run; removes sampling-bias questions from the star-validation. (Not a guardrail violation: the ~60k text-less reviews have nothing to score.)
- **Theming:** sampled ~300–600 stratified negatives (1–2★ with text), batched ~25/batch, run twice. Sampling here is a *design requirement* for the cross-batch stability check, not a compute shortcut — stated as such in markdown.
- **Theming population = all negatives with text (superset), `is_late` attached per review.** One LLM run; the late-only view is a slice. Chart 3 = theme frequency with on-time/late split in one visual (not two co-equal charts — avoids the checklist-tour anti-pattern).

## Smoke-test — resolved offline (see `smoke_test.py`, env pinned in `requirements.txt`)
- **DuckDB** 1.5.4 — standard SQL over dataframes. ✓ Offline SQL seam confirmed.
- **LeIA (Method B)** — installs from PyPI as **`leia-br`** (0.0.1), imports **`from LeIA import SentimentIntensityAnalyzer`**. No git/vendoring needed. Scores accented Portuguese correctly. *Note: lexicon keys on accents — do NOT strip accents before scoring.*
- **HF pt-BR (Method A)** — **`nlptown/bert-base-multilingual-uncased-sentiment`**, torch 2.13.0+cpu / transformers 5.13.0. Emits 1–5 stars → validates directly against the star column. CPU-only here (one-time cached run over ~40k).
- **Anthropic** — ✅ live call verified (`claude-haiku-4-5`, key in gitignored `.env`, loaded via python-dotenv). All 4 walls pass.
- **Env caution:** pandas pinned at **3.0.3** (copy-on-write is default; some 2.x idioms differ) — write against 3.0 semantics.

## Chart 4 — regional cut (locked)
- **Bubble map on lat/long** (Plotly 6.9.0 `scatter_map`, MapLibre — no Mapbox token). Keep `geolocation`, deduped to **city centroids** in silver.
- **color = volume-normalized late-delivery rate** (late ÷ total), **size = order volume** (= confidence), **min-order threshold ~30** to kill small-n noise. Hover → state, late rate, avg review score.
- Rationale: raw density = population map (off-spine, guardrail #8). Two-channel encoding makes "outlier region accounting for volume" honest — big red bubble = high volume *and* high late rate.
- **Next-step (named, not built):** statistical regional outlier detection — **funnel plot** (late rate vs volume with binomial control limits) or empirical-Bayes shrinkage toward the national baseline. The rigorous extension; keeps the notebook spine tight.
- **`geolocation` reinstated** to the spine (reverses the earlier drop) — justified solely by this map.

## Notebook section flow (locked)

Linear reading order of the single notebook. In Zerve these are DAG blocks; order below is narrative/canvas, execution is driven by data dependency.

| # | Section | Tool |
|---|---|---|
| 0 | Title + thesis (one question + arc: quant → sentiment → generative) | md |
| 1 | Smoke-test (4 checks; Zerve re-verifies live Anthropic) | py |
| 2 | Config / caching (`REGENERATE` + `load_or_build`, before any cached step) | py |
| 3 | Bronze — acknowledge 9 tables, scope to spine, name out-of-spine | SQL |
| 4 | Silver — clean/type; **dedup `order_reviews` to unique `review_id`** (814 dupes); items+payments → order grain; `geolocation` → city centroids; **exclude `order_delivered_customer_date IS NULL`** (2,965, report status breakdown) | SQL |
| 5 | Gold — one row per review; delivery facts + order-grain summaries | SQL |
| — | 🥇 Showpiece: SQL/pandas seam (at Gold → feature boundary) | md |
| 6 | EDA + spine framing — kill retention w/ ~97% one-time-buyer fact | pandas |
| 7 | Feature engineering — delivery gap, on-time/late, `is_late`, negative flag | pandas |
| 8 | Chart 1 — review score by delivery outcome | Seaborn |
| 9 | Chart 2 — delivery-gap distribution | Seaborn |
| 10 | Sentiment A/B — nlptown + LeIA over full ~40k text reviews; pickle by `review_id` | py |
| 11 | Sentiment validation — internal agreement vs external validity; inspect disagreement | py/md |
| — | 🥇 Showpiece: sentiment validated against stars | md |
| 12 | Theming — `get_themes` (Anthropic loop), batched, run 2×, defensive parse; pickle; re-join | py |
| — | 🥇 Showpiece: structured output re-joins dataframe | md |
| 13 | Chart 3 — theme frequency ranked, on-time/late split; hover → review | Plotly |
| 14 | Chart 4 — regional bubble map (color=late rate, size=volume, min-n) | Plotly |
| 15 | Limitations / next-steps — 3 exclusions + funnel-plot next-step | md |

**Phase boundary (tighter than the handoff assumed):** because theming moved offline (ADR-0001), the themes pickle exists offline, so Charts 3–4 are also prototyped offline. Phase 2 in Zerve shrinks to: paste SQL → run live, load the two pickles, optionally re-run theming live, re-render.

**Reorder-safety (design property):** section order is cheap to change — Zerve executes a DAG (not line order), and every expensive output is cached via `load_or_build` keyed to `review_id`, so moved cells reload by key without recompute or misalignment. Expensive-to-reverse decisions (grain, exclusions, SQL contract, `review_id` keying) were locked deliberately so they won't be casually restructured. Discipline: cells read by named dataframe/key, never implicit prior-cell state.
