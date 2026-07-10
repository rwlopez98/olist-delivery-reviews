# Phase 2 — Porting to Zerve (step-by-step checklist)

> **The canonical deliverable is [`../zerve_notebook.ipynb`](../zerve_notebook.ipynb)** — 25 cells: narrative markdown + dark-themed code, bronze→silver→gold materialized to disk, and two "how it was built" cells that display the sentiment + theming generation code. Copy cells into a fresh Zerve project. Narrative source lives in [`notebook_narrative.md`](./notebook_narrative.md); this checklist is the porting *rationale + probes*.


Everything was built + verified offline. The port is mechanical, but a few Zerve behaviors
must be *probed* first (marked 🔎) — like the offline smoke-test, discover walls before
building on them. Work top to bottom; verify (✅) after each stage.

Three artifacts cross the boundary: **SQL text**, the **two pickles** (`sentiment.pkl`,
`themes.pkl`), and **pandas/viz code** (`src/olist.py`). Nothing else.

---

## Stage 0 — Pre-flight probes (do these FIRST)

- [ ] **🔎 0.1 How are the 9 tables exposed?** In a Zerve Python block, print what's available.
  Are the CSVs auto-loaded as **pandas DataFrames** (what names?), or only as **SQL tables**,
  or must we read files? → decides Stage 1's approach.
- [ ] **🔎 0.2 What is the SQL block's behavior?** Does a SQL block support multi-statement
  `CREATE VIEW … ; CREATE VIEW …`, or only a single `SELECT`? Does its output become a
  DataFrame variable? → decides native-SQL vs DuckDB-in-Python.
- [ ] **🔎 0.3 pandas version.** `import pandas; print(pandas.__version__)`. If it's < 3.0,
  the pickles (written under 3.0.3) *may* not unpickle → see Stage 3 fallback.
- [ ] **🔎 0.4 File upload.** How do you upload `sentiment.pkl`, `themes.pkl`, and (optionally)
  `src/olist.py` into the workspace? Note the path they land at → set `OLIST_CACHE_DIR`.
- [ ] **🔎 0.5 Anthropic.** `import anthropic` — installed? Is the key available as a Zerve
  secret/env var? (Only needed if you *regenerate* themes; loading the pickle needs neither.)

## Stage 1 — Data layer → `gold` DataFrame

Decision from 0.1/0.2:
- **Path A (recommended, lowest risk): DuckDB inside a Python block.** Reuse `olist.load_gold`
  verbatim — it already runs the exact tested SQL. If tables are DataFrames, register them;
  if the CSVs are uploaded, just set `olist.DATA_PREFIX` to their folder. Runs the SQL live,
  identical results, zero rewrite.
- **Path B: native Zerve SQL blocks** (showcases platform SQL). Split `10/20/30_*.sql` into
  blocks; if views aren't supported, collapse to one CTE query. Swap `read_csv_auto('${DATA_PREFIX}/…')`
  for the live table names.
- [ ] Produce `gold` (one row per review).
- [ ] ✅ Verify: `len(gold) == gold.review_id.nunique()` and spine holds
  (`gold.groupby(is_late).review_score.mean()` ≈ 2.57 late / 4.30 on-time).

## Stage 2 — Analytical layer

- [ ] `df = add_features(gold)` (Python block; paste `add_features`).
- [ ] EDA / retention-kill markdown: 96.9% one-time buyers.
- [ ] Chart 1 — `chart_score_by_outcome(df)` (Seaborn).
- [ ] Chart 2 — `chart_delivery_gap(df)` (Seaborn).
- [ ] ✅ Verify both render.

## Stage 3 — Sentiment

- [ ] Upload `sentiment.pkl`; block: `if REGENERATE_SENTIMENT: build else: load_cache("sentiment.pkl")`.
  Leave `REGENERATE_SENTIMENT = False`.
- [ ] **Fallback if the pickle won't unpickle (pandas mismatch):** re-export as parquet/CSV
  (ask Claude to add that offline), OR set `REGENERATE_SENTIMENT = True` and run in Zerve
  (needs `transformers` + `leia-br` installed — heavier).
- [ ] Validation block: external validity A 75.0% / B 64.3%, internal agree 64.4%; surface
  disagreement examples.
- [ ] ✅ Verify numbers match the offline run.

## Stage 4 — Theming

- [ ] Upload `themes.pkl`; block: `if REGENERATE_THEMES: build_themes(df) else: load_cache("themes.pkl")`.
  Leave `False` for the clean run.
- [ ] (Optional showcase) Flip `REGENERATE_THEMES = True` once to prove the live Anthropic
  loop in-environment (needs `anthropic` + key from 0.5). `get_themes` is the only provider code.
- [ ] Chart 3 — `chart_theme_frequency(themes, df)` (Plotly).
- [ ] ✅ Verify stability note + delivery themes carry the "Late" red.

## Stage 5 — Regional

- [ ] Chart 4 — `chart_regional_bubble(df)` (Plotly). 🔎 If MapLibre tiles are blocked, switch
  `scatter_geo` scope render (already tile-free) — confirm it draws.
- [ ] ✅ Verify SP is large & pale, NE/N are red.

## Stage 6 — Narrative

- [ ] Title/thesis markdown (top); the three 🥇 showpiece callouts; limitations + next-steps.

## Stage 7 — Final verification (definition of done)

- [ ] Run-all from cache (`REGENERATE_* = False`) → completes with **no API key**.
- [ ] All 4 charts render; both showpieces present; excluded rows/tables named with counts.
- [ ] Compare headline numbers to offline (−1.73★; 75.0/64.3; stability 95.7%).

---

### Standing gotchas
- **Zerve env vs offline (probed):** Python 3.11.15, pandas 3.0.3, duckdb 1.5.4 all match — pickles load, tested SQL runs. BUT: `anthropic`/`torch` absent (load both pickles; can't regen sentiment in-Zerve), and **Seaborn is 0.12.x** which crashes on pandas 3.0 (`mode.use_inf_as_na` removed). Fix: `pip install -U "seaborn>=0.13.2"` + restart, or use a matplotlib histogram for Chart 2.
- **Native QUERY blocks don't work here (confirmed by Zerve's AI):** they're thin wrappers over an *external* warehouse (Postgres/Snowflake/…) — no DuckDB connection type, no file access, and the org has zero connections. Zerve's own recommended pattern for CSV→SQL is **DuckDB in a PYTHON block** (`read_csv_auto('olist_*.csv')`), which is what `zerve_notebook.py` does. Alternative (heavier): stand up Postgres, load the CSVs, use native QUERY blocks in Postgres dialect.
- **Native GEN_AI blocks can't batch** — theming must stay the Anthropic-SDK Python loop (ADR-0001).
- **UTF-8**: Portuguese text + `★` need UTF-8 stdout (native in Zerve; only a Windows-console issue offline).
- **Keep accents** for LeIA; **key everything to `review_id`**, never row position.
- **`DATA_PREFIX`** is the single source binding — the only thing Stage 1 rewires.
