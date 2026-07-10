# Olist Delivery → Reviews

A single-notebook analytics showcase answering one question: **does late delivery drive low review scores — and if so, why?** The glossary below fixes the language so markdown, code, and charts all say the same thing.

## Language

**Spine**:
The one thesis the whole notebook serves: late delivery → low review score, and the mechanism behind it. Anything not serving the spine is out of scope.
_Avoid_: story, angle, narrative thread

**Review**:
A single customer review of an order, carrying a 1–5 star score and free Portuguese text. The **unit of analysis** — Gold has one row per Review, keyed by `review_id`. Note `review_id` is only ~99.2% unique in raw data (814 duplicate rows of 99,224); Silver **dedups to a unique `review_id`** so it is a valid grain key and pickle key.
_Avoid_: rating (reserve "rating" for the numeric star score only), feedback

**Order**:
A single customer purchase. Delivery happens per Order. An Order may contain multiple **order items** and multiple **payments**, which is why both are pre-aggregated to Order grain before meeting Review grain.
_Avoid_: purchase, transaction

**Review grain / Order grain**:
"Grain" = the entity one row represents. Gold is at Review grain; delivery facts are Order grain joined onto each Review. Items/payments must be collapsed to Order grain first or delivery lateness double-counts.

**Late delivery**:
An Order whose actual delivery date (`order_delivered_customer_date`) is later than its estimated delivery date (`order_estimated_delivery_date`). Defined only for Orders that were actually delivered — see Excluded orders.
_Avoid_: delayed, slow (use "late" consistently)

**Excluded orders**:
Orders with `order_delivered_customer_date IS NULL` (2,965 orders, ~3%), dropped from the spine because lateness is undefined without an actual delivery date. This is the correct criterion — *not* "canceled status": the null-delivered population spans 7 statuses and does **not** equal the 625 canceled orders (verified). It also catches 8 orders marked `delivered` that carry no delivery date.

**Delivery gap**:
Estimated delivery date − actual delivery date, in days. Positive = early, negative = late. The continuous version of the on-time/late split.

**On-time vs Late**:
The binary delivery outcome derived from Delivery gap, used to split Review scores in Chart 1.

**Disagreement flag**:
A Review where the two sentiment methods (pt-BR transformer vs. LeIA) disagree. These are inspected, never dropped — they are where mixed-sentiment reviews ("great product / terrible delivery") live, which is prime spine evidence.

**Internal agreement vs External validity**:
Two different checks kept explicitly separate. *Internal agreement* = the two sentiment models vs. each other. *External validity* = each model vs. the 1–5 star score (ground truth). Never call the former "confidence."

**Theme**:
An LLM-extracted category describing *why* a negative/late review was negative (e.g. "package never arrived"). Structured output (`theme`, `frequency_rank`, `example_review_id`, optional `severity`) that re-joins the dataframe and drives Chart 3.
_Avoid_: topic, cluster, tag

## Verified data facts (DuckDB, raw CSVs)

- Reviews: 99,224 rows / 98,410 distinct `review_id` / **40,977 with text**.
- One-time buyers: **96.9%** (96,096 customers) — kills the retention angle with evidence.
- Excluded (null delivered-date): **2,965** orders across 7 statuses; ≠ canceled (625). 8 `delivered` orders have no delivery date.
- `geolocation`: ~1M rows → dedup to city centroids for Chart 4 only.
