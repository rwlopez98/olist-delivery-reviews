# =============================================================================
# ZERVE NOTEBOOK — Olist: does late delivery drive low review scores, and why?
#
# Data layer = DuckDB-in-Python (Zerve's own recommended pattern: QUERY blocks need an
# external warehouse; DuckDB in a PYTHON block runs SQL over the CSV files directly).
# The medallion is materialized to disk: each layer runs SQL over the upstream FILES and
# writes its output .parquet to the working dir, so the silver layer is visible on disk and
# a runtime restart never forces a full re-run. [md] = Markdown block; else Python block.
#
# Working dir holds: the 9 Olist CSVs, plus uploaded sentiment.pkl and themes.pkl.
# =============================================================================


# ========== BLOCK 0 · [md] Title =============================================
# # Olist: Does Late Delivery Drive Low Review Scores — and Why?
#
# **The question.** Olist is a Brazilian e-commerce marketplace with ~100K orders and
# customer reviews. Do late deliveries actually drive low review scores — and if so, what
# exactly are customers upset about?
#
# **The answer, up front.** Decisively yes. On-time orders average **4.30★**; late orders
# average **2.57★** — a 1.73-star collapse, with late orders 46% 1-star. And an LLM reading
# the negative reviews shows *why*: the complaints are overwhelmingly about the **parcel**
# (never arrived, arrived late), not the product.
#
# **How it's built — three layers, each cross-validating the next:**
# 1. **SQL · bronze → silver → gold (DuckDB).** Joins and grain resolution to *one row per
#    review*; undelivered orders excluded on the delivery-date null, items and payments
#    pre-aggregated to order grain so lateness is never double-counted.
# 2. **Sentiment · two methods.** A Portuguese transformer and LeIA (Portuguese VADER), each
#    validated against the 1–5★ ground truth (**75% / 64%**) — external validity kept
#    explicitly separate from model-vs-model agreement; disagreements inspected, not dropped.
# 3. **Generative theming · LLM.** Structured output (theme + severity per review) re-joins
#    the dataframe and drives the theme chart; a stability check confirms a review lands on
#    the same theme **96%** of the time across runs.
#
# **Reproducible.** Expensive model outputs are cached and keyed to `review_id`, so the
# notebook runs top-to-bottom from cache with no API calls.
#
# *Data: Olist Brazilian E-Commerce Public Dataset (Kaggle).*


# ========== BLOCK 1 · Setup (all imports + constants + config) ===============
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

sns.set_theme(style="whitegrid")
OUTCOME_ORDER = ["On-time", "Late"]
OUTCOME_PALETTE = {"On-time": "#4c9f70", "Late": "#d1495b"}

# GenAI regeneration: False = load the cached pickle (needs no API key / no torch).
REGENERATE_SENTIMENT = False   # rebuild needs torch (absent in Zerve) — keep False
REGENERATE_THEMES = False      # rebuild needs `pip install anthropic` + a key
print("setup ready · duckdb", duckdb.__version__, "· pandas", pd.__version__)


# ========== BLOCK 2 · [md] Bronze ============================================
# **Bronze** acknowledges all nine auto-loaded CSVs and scopes to the delivery→review
# spine. Out-of-spine (named, unused): `olist_products_dataset`, `olist_sellers_dataset`,
# `product_category_name_translation`. The raw CSVs *are* the bronze layer.


# ========== BLOCK 3 · Bronze (scope) =========================================
BRONZE = {
    "orders":         "olist_orders_dataset",
    "order_reviews":  "olist_order_reviews_dataset",
    "order_items":    "olist_order_items_dataset",
    "order_payments": "olist_order_payments_dataset",
    "customers":      "olist_customers_dataset",
    "geolocation":    "olist_geolocation_dataset",
}
counts = {v: duckdb.sql(f"SELECT COUNT(*) FROM read_csv_auto('{c}.csv')").fetchone()[0]
          for v, c in BRONZE.items()}
print("Bronze scoped to spine:")
for v, n in counts.items():
    print(f"  {v:15} {n:>9,}")


# ========== BLOCK 4 · [md] Silver ============================================
# **Silver** cleans, resolves grain, pre-aggregates — every grain explosion settled here.
# Undelivered orders excluded on the null delivery *date* (not canceled status: the null
# population spans 7 statuses and ≠ canceled). Reviews deduped to one row per `review_id`
# among delivered orders, `order_id` as deterministic tiebreaker (ADR-0002). Items/payments
# → order grain; geolocation → one point per zip. Each table written to .parquet.


# ========== BLOCK 5 · Silver (SQL over CSVs → parquet, shown) ================
con = duckdb.connect()
for view, csv in {"orders": "olist_orders_dataset", "order_reviews": "olist_order_reviews_dataset",
                  "order_items": "olist_order_items_dataset", "order_payments": "olist_order_payments_dataset",
                  "geolocation": "olist_geolocation_dataset"}.items():
    con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{csv}.csv')")
con.execute("""
CREATE TABLE orders_clean AS
  SELECT order_id, customer_id, order_status, order_purchase_timestamp,
         order_delivered_customer_date, order_estimated_delivery_date
  FROM orders WHERE order_delivered_customer_date IS NOT NULL;

CREATE TABLE reviews_dedup AS
  SELECT * EXCLUDE (rn) FROM (
    SELECT rev.*, ROW_NUMBER() OVER (
             PARTITION BY rev.review_id
             ORDER BY (rev.review_comment_message IS NOT NULL) DESC,
                      rev.review_answer_timestamp DESC, rev.order_id) AS rn
    FROM order_reviews rev JOIN orders_clean USING (order_id)) WHERE rn = 1;

CREATE TABLE order_items_agg AS
  SELECT order_id, COUNT(*) n_items, COUNT(DISTINCT seller_id) n_sellers,
         SUM(price) items_total, SUM(freight_value) freight_total
  FROM order_items GROUP BY order_id;

CREATE TABLE order_payments_agg AS
  SELECT order_id, SUM(payment_value) payment_total, COUNT(*) n_payments,
         MAX(payment_installments) max_installments
  FROM order_payments GROUP BY order_id;

CREATE TABLE geolocation_centroid AS
  SELECT geolocation_zip_code_prefix zip_prefix, AVG(geolocation_lat) lat,
         AVG(geolocation_lng) lng, ANY_VALUE(geolocation_state) geo_state
  FROM geolocation GROUP BY geolocation_zip_code_prefix;
""")
for t in ["orders_clean", "reviews_dedup", "order_items_agg", "order_payments_agg", "geolocation_centroid"]:
    con.execute(f"COPY {t} TO '{t}.parquet' (FORMAT parquet)")
    print(f"{t:22} {con.sql(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>8,}  ->  {t}.parquet")
con.close()


# ========== BLOCK 6 · [md] Gold ==============================================
# **Gold** — one row per review, delivery facts joined on, order-grain summaries attached.
# Reads the silver .parquet files; the INNER join to orders_clean enforces the exclusion.
# Written to `gold.parquet` and passed downstream as `gold`.


# ========== BLOCK 7 · Gold (join silver parquets → gold.parquet) =============
gold = duckdb.sql("""
    SELECT r.review_id, r.order_id, r.review_score, r.review_comment_title, r.review_comment_message,
           (r.review_comment_message IS NOT NULL) AS has_text,
           o.order_purchase_timestamp, o.order_delivered_customer_date, o.order_estimated_delivery_date,
           c.customer_unique_id, c.customer_state, c.customer_zip_code_prefix, c.customer_city,
           g.lat AS customer_lat, g.lng AS customer_lng,
           i.n_items, i.n_sellers, i.items_total, i.freight_total,
           p.payment_total, p.max_installments
    FROM 'reviews_dedup.parquet' r
    JOIN 'orders_clean.parquet' o USING (order_id)
    JOIN read_csv_auto('olist_customers_dataset.csv') c USING (customer_id)
    LEFT JOIN 'order_items_agg.parquet' i USING (order_id)
    LEFT JOIN 'order_payments_agg.parquet' p USING (order_id)
    LEFT JOIN 'geolocation_centroid.parquet' g ON c.customer_zip_code_prefix = g.zip_prefix
""").df()
duckdb.sql("COPY (SELECT * FROM gold) TO 'gold.parquet' (FORMAT parquet)")
print(f"gold {len(gold):,} rows · unique review_id: {gold['review_id'].is_unique}  ->  gold.parquet")


# ========== BLOCK 8 · [md] 🥇 The SQL / pandas seam ==========================
# SQL produced the joined, grain-resolved Gold table. Pandas now owns the analytical
# layer — features, sentiment, theming, viz. No column is built twice.


# ========== BLOCK 9 · Features (read gold.parquet → features.parquet) ========
df = duckdb.sql("SELECT * FROM 'gold.parquet'").df()
df["has_text"] = df["has_text"].astype(bool)
delivered = pd.to_datetime(df["order_delivered_customer_date"])
estimated = pd.to_datetime(df["order_estimated_delivery_date"])
df["delivery_gap_days"] = (estimated - delivered).dt.total_seconds() / 86400
df["is_late"] = df["delivery_gap_days"] < 0
df["delivery_outcome"] = df["is_late"].map({False: "On-time", True: "Late"})
df["is_negative"] = df["review_score"] <= 2
duckdb.sql("COPY (SELECT * FROM df) TO 'features.parquet' (FORMAT parquet)")
print(df.groupby("delivery_outcome")["review_score"].mean().round(2).to_dict())
print("late %.1f%% · negative %.1f%%" % (df["is_late"].mean()*100, df["is_negative"].mean()*100))


# ========== BLOCK 10 · [md] EDA — retention is dead ==========================
# 96.9% of customers order exactly once → no retention/CLV story. Surfaced to *close*
# that door with evidence, not open it. The spine stays on delivery→reviews.


# ========== BLOCK 11 · Retention-kill ========================================
pct = duckdb.sql("""
  WITH per AS (SELECT c.customer_unique_id, COUNT(*) n
               FROM read_csv_auto('olist_orders_dataset.csv') o
               JOIN read_csv_auto('olist_customers_dataset.csv') c USING (customer_id)
               GROUP BY 1)
  SELECT ROUND(100.0*COUNT(*) FILTER (WHERE n = 1)/COUNT(*), 1) FROM per
""").fetchone()[0]
print(f"{pct}% of customers order exactly once -> retention angle killed with evidence")


# ========== BLOCK 12 · Chart 1 — score by delivery outcome (Seaborn) =========
df = duckdb.sql("SELECT * FROM 'features.parquet'").df()
prop = (df.groupby("delivery_outcome")["review_score"]
          .value_counts(normalize=True).rename("proportion").reset_index())
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(prop, x="review_score", y="proportion", hue="delivery_outcome",
            hue_order=OUTCOME_ORDER, palette=OUTCOME_PALETTE, ax=ax)
ax.set(xlabel="Review score (stars)", ylabel="Share within outcome")
ax.set_title("Late deliveries collapse into 1-star reviews"); ax.legend(title="Delivery")
plt.show()


# ========== BLOCK 13 · Chart 2 — delivery-gap distribution (matplotlib) =======
# matplotlib, not seaborn.histplot — Zerve's Seaborn 0.12 crashes on pandas 3.0
# (`mode.use_inf_as_na` was removed). Same stacked distribution, no dependency surgery.
gap = df["delivery_gap_days"].clip(-30, 60)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist([gap[~df["is_late"]], gap[df["is_late"]]], bins=np.linspace(-30, 60, 61),
        stacked=True, color=[OUTCOME_PALETTE["On-time"], OUTCOME_PALETTE["Late"]],
        label=OUTCOME_ORDER)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set(xlabel="Delivery gap (days) — positive = early, negative = late", ylabel="Reviews")
ax.set_title("Most deliveries beat the estimate; the late tail is thin but toxic")
ax.legend(title="Delivery"); plt.show()


# ========== BLOCK 14 · Sentiment A/B (load cache) ============================
if REGENERATE_SENTIMENT:
    raise RuntimeError("Sentiment rebuild needs torch (absent in Zerve) — keep REGENERATE_SENTIMENT=False")
sentiment = pd.read_pickle("sentiment.pkl")
print(f"sentiment: {sentiment.shape[0]:,} reviews scored (transformer + LeIA)")


# ========== BLOCK 15 · [md] 🥇 Sentiment validated against stars =============
# *Internal agreement* (models vs. each other) and *external validity* (each vs. the star
# ground truth) do different jobs — reported separately. Disagreement rows are inspected,
# never dropped: they are the mixed-sentiment reviews ("great product / terrible delivery").


# ========== BLOCK 16 · Sentiment validation + disagreements ==================
print("A external validity: %.1f%%" % (sentiment["a_valid"].mean()*100))
print("B external validity: %.1f%%" % (sentiment["b_valid"].mean()*100))
print("A-B internal agree : %.1f%%" % (sentiment["ab_agree"].mean()*100))
print("disagreement rows  :", int((~sentiment["ab_agree"]).sum()))
txt = duckdb.sql("SELECT review_id, review_comment_message FROM 'features.parquet'").df()
disagree = sentiment[~sentiment["ab_agree"]].merge(txt, on="review_id")
print("\nmixed-sentiment examples (models split):")
for _, r in disagree.sample(3, random_state=1).iterrows():
    print(f"  {r['review_score']}star | A={r['label_a']:>8} B={r['label_b']:>8} | "
          f"{str(r['review_comment_message'])[:68]}")


# ========== BLOCK 17 · Theming (load cache) ==================================
if REGENERATE_THEMES:
    raise RuntimeError("Theming rebuild needs `pip install anthropic` + ANTHROPIC_API_KEY")
themes = pd.read_pickle("themes.pkl")
print(themes["theme"].value_counts().to_dict())


# ========== BLOCK 18 · [md] 🥇 Structured output re-joins the dataframe =======
# The LLM themes (structured: theme, severity, review_id) re-join Gold and drive Chart 3.
# Batching is forced by the context window; an offline stability check found a review lands
# on the same theme 95.7% of the time across runs — the answer to "did it hallucinate?"


# ========== BLOCK 19 · Chart 3 — theme frequency, on-time/late split (Plotly) =
df = duckdb.sql("SELECT * FROM 'features.parquet'").df()
m = themes.merge(df[["review_id", "is_late"]], on="review_id", how="left")
m["delivery_outcome"] = m["is_late"].map({False: "On-time", True: "Late"})
order = m["theme"].value_counts().index.tolist()[::-1]
plot = m.groupby(["theme", "delivery_outcome"]).size().reset_index(name="n")
fig = px.bar(plot, x="n", y="theme", color="delivery_outcome", orientation="h",
             category_orders={"theme": order, "delivery_outcome": OUTCOME_ORDER},
             color_discrete_map=OUTCOME_PALETTE,
             title="Why negative reviews are negative — delivery themes dominate",
             labels={"n": "Reviews", "theme": "", "delivery_outcome": "Delivery"})
fig.show()


# ========== BLOCK 20 · Chart 4 — regional bubble map (Plotly) ================
df = duckdb.sql("SELECT * FROM 'features.parquet'").df()
agg = (df.groupby("customer_city")
         .agg(n=("review_id", "size"), late_rate=("is_late", "mean"),
              avg_score=("review_score", "mean"), lat=("customer_lat", "mean"),
              lng=("customer_lng", "mean"), state=("customer_state", "first"))
         .reset_index())
agg = agg[(agg["n"] >= 30) & agg["lat"].notna()]
fig = px.scatter_geo(agg, lat="lat", lon="lng", size="n", color="late_rate",
                     color_continuous_scale="Reds", scope="south america", size_max=38,
                     hover_name="customer_city",
                     hover_data={"state": True, "late_rate": ":.1%", "avg_score": ":.2f",
                                 "n": True, "lat": False, "lng": False},
                     labels={"late_rate": "Late rate", "n": "Orders"},
                     title="Where late deliveries concentrate (cities with ≥30 orders)")
fig.update_geos(fitbounds="locations", showcountries=True)
fig.show()


# ========== BLOCK 21 · [md] Limitations & next steps =========================
# - **Predictive modeling left out deliberately** — the spine is explanatory, not predictive.
# - **Seller segmentation stays a supporting cut** — doesn't compound with the delivery story.
# - **BI dashboarding out of scope** for a notebook deliverable (named because the JD lists it).
# - **Next step:** statistical regional outlier detection (funnel plot / empirical-Bayes
#   shrinkage) to flag states significantly worse than baseline given their volume.
