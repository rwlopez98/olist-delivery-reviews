# =============================================================================
# ZERVE NOTEBOOK — Olist: does late delivery drive low review scores, and why?
#
# One block per section (delimited below). Copy each block into a Zerve block in
# order. The medallion is materialized to disk: bronze/silver/gold live as tables
# in `olist.duckdb`, each silver + gold table is also written to .parquet, and the
# pandas layer reads those files — nothing critical is memory-only, so a runtime
# restart never forces a full re-run. [md] blocks are Markdown; the rest are Python.
#
# Prereq: the 9 Olist CSVs sit in the working dir (bare filenames resolve there).
# Uploaded artifacts also in the working dir: sentiment.pkl, themes.pkl.
# =============================================================================


# ========== BLOCK 0 · [md] Title =============================================
# # Olist: Does Late Delivery Drive Low Review Scores — and Why?
# **Thesis:** late delivery → low review score, and an LLM explains the mechanism.
# **Arc:** SQL proves late→bad (quantitative) → sentiment quantifies it → an LLM
# themes the negative reviews (generative). Bronze→Silver→Gold is materialized on disk.


# ========== BLOCK 1 · Setup (all imports + constants + config) ===============
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")
OUTCOME_ORDER = ["On-time", "Late"]
OUTCOME_PALETTE = {"On-time": "#4c9f70", "Late": "#d1495b"}

DB = "olist.duckdb"   # persistent medallion store (bronze/silver/gold tables)

# GenAI regeneration: False = load the cached pickle (needs no API key / no torch).
REGENERATE_SENTIMENT = False   # sentiment needs torch to rebuild — keep False in Zerve
REGENERATE_THEMES = False      # theming needs `pip install anthropic` + a key to rebuild
print("setup ready · duckdb", duckdb.__version__, "· pandas", pd.__version__)


# ========== BLOCK 2 · [md] Bronze ============================================
# **Bronze** acknowledges all nine auto-loaded tables and scopes to the delivery→review
# spine. Out-of-spine (named, unused): `olist_products_dataset`, `olist_sellers_dataset`,
# `product_category_name_translation`.


# ========== BLOCK 3 · Bronze (scope raw tables as views) =====================
BRONZE = {
    "orders":         "olist_orders_dataset",
    "order_reviews":  "olist_order_reviews_dataset",
    "order_items":    "olist_order_items_dataset",
    "order_payments": "olist_order_payments_dataset",
    "customers":      "olist_customers_dataset",
    "geolocation":    "olist_geolocation_dataset",
}
con = duckdb.connect(DB)
for view, csv in BRONZE.items():
    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_csv_auto('{csv}.csv')")
print("Bronze scoped to spine:", ", ".join(BRONZE))
con.close()


# ========== BLOCK 4 · [md] Silver ============================================
# **Silver** cleans, resolves the grain, and pre-aggregates — every grain explosion is
# settled here. Undelivered orders are excluded on the null delivery *date* (not canceled
# status: the null population spans 7 statuses and ≠ canceled). Reviews are deduped to one
# row per `review_id` among delivered orders, with `order_id` as a deterministic tiebreaker
# (ADR-0002). Items/payments collapse to order grain; geolocation to one point per zip.


# ========== BLOCK 5 · Silver (materialize + show each table) =================
con = duckdb.connect(DB)
con.execute("""
CREATE OR REPLACE TABLE orders_clean AS
  SELECT order_id, customer_id, order_status, order_purchase_timestamp,
         order_delivered_customer_date, order_estimated_delivery_date
  FROM orders WHERE order_delivered_customer_date IS NOT NULL;

CREATE OR REPLACE TABLE reviews_dedup AS
  SELECT * EXCLUDE (rn) FROM (
    SELECT rev.*, ROW_NUMBER() OVER (
             PARTITION BY rev.review_id
             ORDER BY (rev.review_comment_message IS NOT NULL) DESC,
                      rev.review_answer_timestamp DESC, rev.order_id) AS rn
    FROM order_reviews rev JOIN orders_clean USING (order_id)) WHERE rn = 1;

CREATE OR REPLACE TABLE order_items_agg AS
  SELECT order_id, COUNT(*) n_items, COUNT(DISTINCT seller_id) n_sellers,
         SUM(price) items_total, SUM(freight_value) freight_total
  FROM order_items GROUP BY order_id;

CREATE OR REPLACE TABLE order_payments_agg AS
  SELECT order_id, SUM(payment_value) payment_total, COUNT(*) n_payments,
         MAX(payment_installments) max_installments
  FROM order_payments GROUP BY order_id;

CREATE OR REPLACE TABLE geolocation_centroid AS
  SELECT geolocation_zip_code_prefix zip_prefix, AVG(geolocation_lat) lat,
         AVG(geolocation_lng) lng, ANY_VALUE(geolocation_state) geo_state
  FROM geolocation GROUP BY geolocation_zip_code_prefix;
""")
for t in ["orders_clean", "reviews_dedup", "order_items_agg",
          "order_payments_agg", "geolocation_centroid"]:
    con.execute(f"COPY {t} TO '{t}.parquet' (FORMAT parquet)")
    n = con.sql(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t:22} {n:>8,} rows  ->  {t}.parquet")
con.close()


# ========== BLOCK 6 · [md] Gold ==============================================
# **Gold** — one row per review, delivery facts joined on, order-grain summaries attached.
# The INNER join to `orders_clean` enforces the exclusion; review text is kept for the
# sentiment + theming layers. Written to `gold.parquet` for the pandas layer.


# ========== BLOCK 7 · Gold (materialize + verify grain) ======================
con = duckdb.connect(DB)
con.execute("""
CREATE OR REPLACE TABLE gold_reviews AS
  SELECT r.review_id, r.order_id, r.review_score, r.review_comment_title, r.review_comment_message,
         (r.review_comment_message IS NOT NULL) AS has_text,
         o.order_purchase_timestamp, o.order_delivered_customer_date, o.order_estimated_delivery_date,
         c.customer_unique_id, c.customer_state, c.customer_zip_code_prefix, c.customer_city,
         g.lat AS customer_lat, g.lng AS customer_lng,
         i.n_items, i.n_sellers, i.items_total, i.freight_total,
         p.payment_total, p.max_installments
  FROM reviews_dedup r
  JOIN orders_clean o USING (order_id)
  JOIN customers c USING (customer_id)
  LEFT JOIN order_items_agg i USING (order_id)
  LEFT JOIN order_payments_agg p USING (order_id)
  LEFT JOIN geolocation_centroid g ON c.customer_zip_code_prefix = g.zip_prefix;
""")
con.execute("COPY gold_reviews TO 'gold.parquet' (FORMAT parquet)")
n, u = con.sql("SELECT COUNT(*), COUNT(DISTINCT review_id) FROM gold_reviews").fetchone()
print(f"gold_reviews {n:,} rows · unique review_id: {n == u}  ->  gold.parquet")
con.close()


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
con = duckdb.connect(DB)
pct = con.sql("""
  WITH per AS (SELECT c.customer_unique_id, COUNT(*) n
               FROM orders o JOIN customers c USING (customer_id) GROUP BY 1)
  SELECT ROUND(100.0*COUNT(*) FILTER (WHERE n = 1)/COUNT(*), 1) FROM per
""").fetchone()[0]
con.close()
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


# ========== BLOCK 13 · Chart 2 — delivery-gap distribution ===================
# matplotlib (not seaborn.histplot) — Zerve's Seaborn 0.12 crashes on pandas 3.0
# (`mode.use_inf_as_na` was removed). Same stacked distribution, no dependency surgery.
import numpy as np
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
# Batching is forced by the context window; a stability check (offline) found a review
# lands on the same theme 95.7% of the time across runs — the answer to "did it hallucinate?"


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
