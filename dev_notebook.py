# %% [markdown]
# # Olist: Does Late Delivery Drive Low Review Scores — and Why?
#
# **Thesis:** late delivery → low review score, and an LLM explains the mechanism.
# **Arc:** SQL proves late→bad (quantitative) → sentiment quantifies it → LLM themes the
# negative reviews (generative).
#
# This is the OFFLINE development notebook. Logic lives in `src/olist.py` so each function
# ports into a Zerve block unchanged; here we orchestrate it top-to-bottom. Expensive
# outputs (sentiment, themes) load from `data/cache/*.pkl` — a plain run-all hits no API.

# %%
import sys; sys.path.insert(0, "src")
import olist

# --- Config -----------------------------------------------------------------
# GenAI regeneration switches: False loads the cached pickle (reviewer default — a plain
# run-all hits no API); True re-runs against the model. Per-task because sentiment is a
# ~15-min CPU job while theming is a ~2-min API job — regenerate one without the other.
REGENERATE_SENTIMENT = False
REGENERATE_THEMES = False
# Data source location — single prefix; reassign to relocate the CSVs or embed into Zerve.
# olist.DATA_PREFIX = "data/raw"   # (default; override here if the layout differs)

# %% [markdown]
# ## Environment smoke-test
# See `smoke_test.py` — DuckDB, LeIA, the pt-BR transformer, and the Anthropic call are all
# verified before anything is built on them.

# %% [markdown]
# ## Bronze → Silver → Gold (SQL)
# SQL owns the data layer. `load_gold()` runs `sql/10_bronze → 20_silver → 30_gold`:
# scope to the spine tables, resolve grain (dedup reviews to a unique `review_id` among
# delivered orders — ADR-0002), pre-aggregate items/payments to order grain, exclude
# undelivered orders (`order_delivered_customer_date IS NULL`). Gold = one row per review.
# %%
gold = olist.load_gold()
print("gold:", gold.shape, "| unique review_id:", gold["review_id"].is_unique)

# %% [markdown]
# ### 🥇 The SQL / pandas seam
# SQL produced the joined, grain-resolved table. Pandas now owns the analytical layer —
# no column is built twice.

# %% [markdown]
# ## EDA — and killing the retention angle with evidence
# 96.9% of customers order exactly once, so there is no retention/CLV story to tell here.
# We surface that to *close* the door, not open it. The spine stays on delivery→reviews.
# %%
df = olist.add_features(gold)
print("late rate: %.1f%% | negative rate: %.1f%%" %
      (df["is_late"].mean() * 100, df["is_negative"].mean() * 100))
print(df.groupby("delivery_outcome")["review_score"].mean().round(2).to_dict())

# %% [markdown]
# ## Chart 1 — review score by delivery outcome (Seaborn)
# %%
olist.chart_score_by_outcome(df, "reports/figures/chart1_score_by_outcome.png")

# %% [markdown]
# ## Chart 2 — delivery-gap distribution (Seaborn)
# %%
olist.chart_delivery_gap(df, "reports/figures/chart2_delivery_gap.png")

# %% [markdown]
# ## Sentiment A/B — pt-BR transformer vs. LeIA
# Two architectures (transformer + lexicon) over all ~38k text reviews; cached by review_id.
# %%
if REGENERATE_SENTIMENT:
    sentiment = olist.save_cache("sentiment.pkl", olist.build_sentiment(df))
else:
    sentiment = olist.load_cache("sentiment.pkl")

# %% [markdown]
# ### 🥇 Sentiment validated against stars
# *Internal agreement* (models vs. each other) and *external validity* (each vs. the star
# ground truth) do different jobs — reported separately. Disagreement rows are inspected,
# never dropped: they are the mixed-sentiment reviews ("great product / terrible delivery").
# %%
print("A external validity: %.1f%%" % (sentiment["a_valid"].mean() * 100))
print("B external validity: %.1f%%" % (sentiment["b_valid"].mean() * 100))
print("A–B internal agree : %.1f%%" % (sentiment["ab_agree"].mean() * 100))
print("disagreement rows  :", int((~sentiment["ab_agree"]).sum()))

# Surface the disagreement rows — these are the mixed-sentiment reviews, prime evidence.
disagree = (sentiment[~sentiment["ab_agree"]]
            .merge(df[["review_id", "review_comment_message"]], on="review_id"))
print("\nsample mixed-sentiment reviews (models split):")
for _, r in disagree.sample(3, random_state=1).iterrows():
    print(f"  ★{r['review_score']} | A={r['label_a']:>8} B={r['label_b']:>8} | "
          f"{str(r['review_comment_message'])[:70]}")

# %% [markdown]
# ## Theming — the generative layer
# An LLM reads batches of negative reviews and returns structured output (theme, severity)
# keyed to review_id. Batching is forced by the context window. Cached by review_id.
# %%
if REGENERATE_THEMES:
    themes = olist.save_cache("themes.pkl", olist.build_themes(df))
else:
    themes = olist.load_cache("themes.pkl")
print(themes["theme"].value_counts().to_dict())

# %% [markdown]
# ### 🥇 Structured output re-joins the dataframe → Chart 3 (Plotly)
# The themes re-join Gold and drive the chart; each bar splits on-time vs. late in one
# visual. Stability check (`olist.theme_stability`) reports how often a review lands on the
# same theme across runs — the answer to "how do you know it didn't hallucinate."
# %%
olist.chart_theme_frequency(themes, df, save_html="reports/figures/chart3_themes.html",
                            save_png="reports/figures/chart3_themes.png")

# %% [markdown]
# ## Chart 4 — where late deliveries concentrate (Plotly)
# City bubble map: size = order volume, color = volume-normalized late rate (never raw
# counts, or it becomes a population map). São Paulo is huge but pale; the red outliers sit
# far from the seller hub.
# %%
olist.chart_regional_bubble(df, save_html="reports/figures/chart4_regional.html",
                            save_png="reports/figures/chart4_regional.png")

# %% [markdown]
# ## Limitations & next steps
# - **Predictive modeling left out deliberately** — the spine is explanatory, not predictive.
# - **Seller segmentation stays a supporting cut** — it doesn't compound with the delivery story.
# - **BI dashboarding is out of scope** for a notebook deliverable (named because the JD lists it).
# - **Next step:** statistical regional outlier detection (funnel plot / empirical-Bayes
#   shrinkage) to flag states significantly worse than baseline given their volume.
