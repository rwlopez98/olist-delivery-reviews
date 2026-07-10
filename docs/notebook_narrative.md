# Notebook narrative — Markdown blocks for the Zerve canvas

Paste each section below into a **Markdown block** at the indicated position. Convention:
every block leads with the *finding or decision*; the tool rides second. The two
"How it was built" blocks display the generation code for the cached pickles.

---

## [M0] Title block — top of notebook

# Olist: Does Late Delivery Drive Low Review Scores — and Why?

**The question.** Olist is a Brazilian e-commerce marketplace with ~100K orders and
customer reviews. Do late deliveries actually drive low review scores — and if so, what
exactly are customers upset about?

**The answer, up front.** Decisively yes. On-time orders average **4.30★**; late orders
average **2.57★** — a 1.73-star collapse, with late orders 46% 1-star. And an LLM reading
the negative reviews shows *why*: the complaints are overwhelmingly about the **parcel**
(never arrived, arrived late), not the product.

**How it's built — three layers, each cross-validating the next:**
1. **SQL · bronze → silver → gold (DuckDB).** Joins and grain resolution to *one row per
   review*; undelivered orders excluded, items/payments pre-aggregated to order grain.
2. **Sentiment · two methods.** A Portuguese transformer and LeIA (Portuguese VADER), each
   validated against the 1–5★ ground truth — external validity kept separate from
   model-vs-model agreement; disagreements inspected, not dropped.
3. **Generative theming · LLM.** Structured output (theme + severity per review) re-joins
   the dataframe and drives the theme chart; a stability check guards against hallucination.

**Reproducible.** Expensive model outputs are cached and keyed to `review_id`, so the
notebook runs top-to-bottom from cache with no API calls.

*Data: Olist Brazilian E-Commerce Public Dataset (Kaggle).*

---

## [M1] Bronze — before `bronze_table_row_counts`

### 1 · Bronze — scope the raw data
The dataset ships **nine** tables. This analysis follows a single spine — *delivery →
reviews* — so Bronze scopes to the six that serve it (`orders`, `order_reviews`,
`order_items`, `order_payments`, `customers`, `geolocation`) and names the three it
deliberately sets aside (`products`, `sellers`, `product_category_name_translation`).
Nothing is silently dropped.

---

## [M2] Silver — before `duckdb_etl_pipeline`

### 2 · Silver — clean, resolve grain, pre-aggregate
Every grain trap is settled here, so the review grain downstream is never corrupted:

- **Exclusions.** Orders with no delivery date are removed — lateness is undefined without
  one. The criterion is the null *date*, not "canceled" status: verified, because the 2,965
  null-date orders span seven statuses and don't equal the 625 canceled ones.
- **Review grain.** `review_id` isn't unique in the raw data (814 duplicate rows, and some
  IDs map to multiple orders), so reviews are deduped to one row per `review_id` among
  delivered orders, with `order_id` as a deterministic tiebreaker.
- **Order-grain aggregation.** Items and payments are collapsed to one row per order
  *before* they meet the review grain — otherwise a 5-item order would count its delivery
  lateness five times and every average would quietly break.

Each silver table is written to Parquet, so the medallion is inspectable on disk.

---

## [M3] Gold — before `duckdb_gold_table_export`

### 3 · Gold — one row per review
The analytical table: **one row per review**, delivery facts joined on, order-grain
summaries attached. The inner join to `orders_clean` is what enforces the exclusion.
Result: 95,645 rows, unique on `review_id`.

> **🥇 The SQL / pandas seam.** SQL owns the data layer — set-based joins and grain
> resolution, structured bronze → silver → gold. Pandas owns the analytical layer — feature
> engineering, sentiment, theming, visualization. The split is by strength, and no table is
> ever built twice.

---

## [M4] Features & EDA — before `delivery_features_engineering`

### 4 · Features, and a hypothesis killed with evidence
Pandas derives the continuous **delivery gap** (estimated − actual, in days), the binary
**on-time/late** flag, and a **negative-review** flag (1–2★).

Before going further, one tempting angle is closed with evidence: **96.9% of customers order
exactly once.** There is no retention or lifetime-value story in this data — surfaced here
to *rule it out*, so the spine stays on delivery.

---

## [M5] Quantitative signal — before `review_score_by_delivery`

### 5 · The quantitative signal
Two charts establish late→bad on hard numbers, before any NLP. Chart 1 splits the review-
score distribution by delivery outcome; Chart 2 shows the delivery-gap distribution. Olist
pads its estimates (median ~12 days early), so the damage lives in the thin tail that
crosses zero into "late."

---

## [M6] Sentiment — before `load_sentiment_scores`

### 6 · Sentiment — two methods, validated against ground truth
Reviews are Portuguese free text. Two **independent** methods score them:

- **Method A — a Portuguese-native transformer** (`nlptown/bert-base-multilingual-uncased-sentiment`):
  reads Portuguese directly, emits a 1–5 star prediction.
- **Method B — LeIA**, a Portuguese adaptation of VADER: a lexicon — a deliberately
  *different architecture* from the transformer, so the two act as independent checks.
  (Stock English VADER returns near-zero garbage on Portuguese and is not used.)

**How it was built** (generated offline — the transformer needs `torch`; cached to
`sentiment.pkl` keyed to `review_id`):

```python
# Method A — Portuguese transformer → 1–5 stars
from transformers import pipeline
clf = pipeline("sentiment-analysis",
               model="nlptown/bert-base-multilingual-uncased-sentiment",
               truncation=True, max_length=512)
star_a = [int(r["label"].split()[0]) for r in clf(texts, batch_size=64)]

# Method B — LeIA (Portuguese VADER), lexicon-based
from LeIA import SentimentIntensityAnalyzer
leia = SentimentIntensityAnalyzer()
compound_b = [leia.polarity_scores(t)["compound"] for t in texts]

# Map both models AND the star ground truth to one neg/neu/pos scale, then compare
def stars_to_polarity(s):    return "negative" if s <= 2 else "neutral" if s == 3 else "positive"
def compound_to_polarity(c): return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

out = pd.DataFrame({"review_id": review_ids, "star_a": star_a,
                    "compound_b": compound_b, "review_score": scores})
out["label_a"]     = out["star_a"].map(stars_to_polarity)
out["label_b"]     = out["compound_b"].map(compound_to_polarity)
out["label_truth"] = out["review_score"].map(stars_to_polarity)
out["ab_agree"] = out["label_a"] == out["label_b"]      # internal agreement (A vs B)
out["a_valid"]  = out["label_a"] == out["label_truth"]  # external validity (A vs stars)
out["b_valid"]  = out["label_b"] == out["label_truth"]  # external validity (B vs stars)
```

---

## [M7] Sentiment validity — before `sentiment_validity_analysis`

> **🥇 Validated against stars, not against each other.** Two checks, kept explicitly
> separate. *External validity* is each model vs. the 1–5★ ground truth (A **75.0%**,
> B **64.3%**). *Internal agreement* is the models vs. each other (**64.4%**). They answer
> different questions — model-vs-model agreement is **not** "confidence." The ~13,800 rows
> where the models disagree are **inspected, not dropped**: that is where mixed-sentiment
> reviews live ("great product / terrible delivery"), the sharpest evidence for the spine.

---

## [M8] Theming — before `load_themes_count`

### 7 · Generative theming — what customers are actually angry about
Sentiment says *how* negative a review is; it doesn't say *why*. An LLM reads batches of
negative reviews and returns **structured output** — one theme + severity per review — from
a fixed vocabulary so labels stay comparable across batches. The structured result re-joins
the dataframe and drives the theme chart.

**How it was built** (runs against the Anthropic API — not available in this environment;
cached to `themes.pkl` keyed to `review_id`). This is the *one* provider-specific function,
with defensive JSON parsing:

```python
THEME_VOCAB = ["never_arrived", "late_delivery", "wrong_or_missing_item", "damaged_product",
               "not_as_described", "poor_quality", "refund_or_billing", "other"]

def get_themes(review_batch, model="claude-haiku-4-5"):
    """Theme one batch of negative reviews -> [{review_id, theme, severity}].
    The ONLY provider-specific code; everything downstream is provider-agnostic."""
    prompt = build_prompt(THEME_VOCAB, review_batch)     # asks for a JSON array, one row/review
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(2):                             # one retry on malformed JSON
        text = client.messages.create(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]).content[0].text
        try:
            rows = parse_json_array(text)                # strips ```fences```, grabs outer [...]
            for r in rows:                               # clamp unknown labels to "other"
                if r["theme"] not in THEME_VOCAB: r["theme"] = "other"
            return rows
        except Exception:
            if attempt == 1: raise

# Batching is forced by the context window; sampling is required for the stability check.
themes = pd.concat([pd.DataFrame(get_themes(b)) for b in batches(negatives, size=25)])
```

---

## [M9] Structured output — before `delivery_themes_review_analysis`

> **🥇 Structured output re-joins the dataframe.** The AI-assisted analytics workflow shown
> literally: the LLM's structured output re-joins Gold by `review_id` and drives the chart
> below — it never dead-ends as a prose blob. Verification is **stability**: run the same
> sample twice and a review lands on the same theme **95.7%** of the time — the answer to
> "how do you know it didn't hallucinate." The chart splits each theme on-time vs. late in a
> single frame: the delivery themes (`never_arrived`, `late_delivery`) carry almost all the
> "Late" mass, while product complaints sit on on-time orders.

---

## [M10] Regional — before `delivery_lateness_map`

### 8 · Where lateness concentrates
The map colors each city by its **volume-normalized late rate** (never raw counts, which
would just reproduce population) and sizes it by order volume. São Paulo — the seller hub —
is huge but pale; the red outliers sit far up the North/Northeast coast, where deliveries
travel farthest from the sellers. Cities under 30 orders are dropped so small-sample noise
doesn't masquerade as an outlier.

---

## [M11] Limitations — end of notebook

### Limitations & next steps
- **Predictive modeling left out deliberately** — the spine is explanatory, not predictive.
- **Seller segmentation stays a supporting cut** — it doesn't compound with the delivery story.
- **BI dashboarding out of scope** for a notebook deliverable (named because the JD lists it).
- **Next step:** statistical regional outlier detection — a **funnel plot** (late rate vs.
  volume with binomial control limits) or empirical-Bayes shrinkage — to flag states
  significantly worse than baseline given their volume.
