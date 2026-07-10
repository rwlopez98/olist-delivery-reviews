"""
Core logic for the Olist Delivery → Reviews notebook.

Each function is self-contained so it ports into a Zerve block unchanged. Offline
orchestration lives in `dev_notebook.py`; the expensive outputs (sentiment, themes)
are cached to `data/cache/*.pkl` keyed to `review_id`.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path

import pandas as pd

# Repo root, so paths work whether called from a script or a notebook cell.
ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Pathing — one prefix per location so the project embeds cleanly into Zerve.
# Override via env var (or reassign the attribute) when the folder layout differs.
# In Zerve you either point DATA_PREFIX at wherever the CSVs live, or drop the Bronze
# block entirely and read the auto-loaded live tables.
# --------------------------------------------------------------------------- #
DATA_PREFIX = Path(os.environ.get("OLIST_DATA_PREFIX", ROOT / "data" / "raw"))
SQL_DIR = Path(os.environ.get("OLIST_SQL_DIR", ROOT / "sql"))
CACHE_DIR = Path(os.environ.get("OLIST_CACHE_DIR", ROOT / "data" / "cache"))


# --------------------------------------------------------------------------- #
# Reproducibility — explicit cache primitives. Notebook cells wrap the GenAI steps
# in `if REGENERATE_X: build+save else: load` so the control flow is visible.
# --------------------------------------------------------------------------- #
def save_cache(name: str, obj):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / name, "wb") as f:
        pickle.dump(obj, f)
    return obj


def load_cache(name: str):
    with open(CACHE_DIR / name, "rb") as f:
        return pickle.load(f)


def load_or_build(cache_name: str, build_fn, regenerate: bool = False):
    """Convenience for scripts: build+cache when regenerating or when cache is missing
    (a missing cache must fall through, not crash); otherwise load."""
    if regenerate or not (CACHE_DIR / cache_name).exists():
        return save_cache(cache_name, build_fn())
    return load_cache(cache_name)


# --------------------------------------------------------------------------- #
# Data layer — run the SQL, return Gold (one row per review)
# --------------------------------------------------------------------------- #
def load_gold(con=None) -> pd.DataFrame:
    """Execute bronze→silver→gold SQL in DuckDB and return the Gold dataframe.

    In Zerve this whole function collapses to a single query against the live
    `gold_reviews` view; offline it materializes the CSV-backed views first.
    """
    import duckdb
    con = con or duckdb.connect()
    prefix = str(DATA_PREFIX).replace("\\", "/")  # DuckDB prefers forward slashes
    for sql_file in ["10_bronze.sql", "20_silver.sql", "30_gold.sql"]:
        sql = (SQL_DIR / sql_file).read_text(encoding="utf-8")
        sql = sql.replace("${DATA_PREFIX}", prefix)  # only Bronze contains the token
        con.execute(sql)
    return con.sql("SELECT * FROM gold_reviews").df()


# --------------------------------------------------------------------------- #
# Feature engineering (pandas owns the analytical layer)
# --------------------------------------------------------------------------- #
def add_features(gold: pd.DataFrame) -> pd.DataFrame:
    """Derive the spine's analytical columns from Gold's raw delivery dates + score."""
    df = gold.copy()
    # has_text may arrive as bool (DuckDB) or 0/1 int (portable Zerve SQL) — normalize.
    df["has_text"] = df["has_text"].astype(bool)
    delivered = pd.to_datetime(df["order_delivered_customer_date"])
    estimated = pd.to_datetime(df["order_estimated_delivery_date"])

    # Positive = delivered early, negative = late. The continuous form of the outcome.
    df["delivery_gap_days"] = (estimated - delivered).dt.total_seconds() / 86400
    df["is_late"] = df["delivery_gap_days"] < 0
    df["delivery_outcome"] = df["is_late"].map({False: "On-time", True: "Late"})
    # 1-2 stars = negative; this defines the theming population.
    df["is_negative"] = df["review_score"] <= 2
    return df


# --------------------------------------------------------------------------- #
# Charts 1–2 (Seaborn) — the quantitative spine
# --------------------------------------------------------------------------- #
OUTCOME_ORDER = ["On-time", "Late"]
OUTCOME_PALETTE = {"On-time": "#4c9f70", "Late": "#d1495b"}


def chart_score_by_outcome(df: pd.DataFrame, save_path=None):
    """Chart 1: review-score distribution, split by delivery outcome.

    Proportions are taken *within* each outcome so the on-time majority doesn't
    swamp the smaller late group — the shape comparison is the point.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    prop = (df.groupby("delivery_outcome")["review_score"]
              .value_counts(normalize=True).rename("proportion").reset_index())
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(prop, x="review_score", y="proportion", hue="delivery_outcome",
                hue_order=OUTCOME_ORDER, palette=OUTCOME_PALETTE, ax=ax)
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Share of reviews within outcome")
    ax.set_title("Late deliveries collapse into 1-star reviews")
    ax.legend(title="Delivery")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def chart_delivery_gap(df: pd.DataFrame, save_path=None, clip=(-30, 60)):
    """Chart 2: distribution of the delivery gap (estimated − actual, days).

    Clipped to a readable window; 0 is the on-time/late boundary. Most orders land
    well early — the late tail (left of 0) is where the bad reviews concentrate.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df.assign(gap=df["delivery_gap_days"].clip(*clip))
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(d, x="gap", hue="delivery_outcome", hue_order=OUTCOME_ORDER,
                 palette=OUTCOME_PALETTE, bins=60, multiple="stack", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Delivery gap (days) — positive = early, negative = late")
    ax.set_ylabel("Reviews")
    ax.set_title("Most deliveries beat the estimate; the late tail is thin but toxic")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


# --------------------------------------------------------------------------- #
# Sentiment A/B — two methods, different architectures
# --------------------------------------------------------------------------- #
# 1-2 stars = negative, 3 = neutral, 4-5 = positive. Shared mapping so both models
# and the ground-truth star score speak the same three-way language.
def stars_to_polarity(star) -> str:
    return "negative" if star <= 2 else ("neutral" if star == 3 else "positive")


def compound_to_polarity(c, pos=0.05, neg=-0.05) -> str:
    return "positive" if c >= pos else ("negative" if c <= neg else "neutral")


def method_a_transformer(texts, batch_size=64):
    """Method A: nlptown multilingual product-review transformer → 1-5 star integer."""
    from transformers import pipeline
    clf = pipeline("sentiment-analysis",
                   model="nlptown/bert-base-multilingual-uncased-sentiment",
                   truncation=True, max_length=512, device=-1)
    out = clf(list(texts), batch_size=batch_size)
    return [int(r["label"].split()[0]) for r in out]  # "4 stars" -> 4


def method_b_leia(texts):
    """Method B: LeIA (Portuguese VADER), lexicon-based → compound score in [-1, 1]."""
    from LeIA import SentimentIntensityAnalyzer
    s = SentimentIntensityAnalyzer()
    return [s.polarity_scores(t)["compound"] for t in texts]


def build_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run both methods over reviews WITH text; return a frame keyed to review_id.

    Keyed to review_id (never row position) so it re-joins Gold safely after any
    reindex on the Zerve side. Cache this via load_or_build('sentiment.pkl', ...).
    """
    sub = df.loc[df["has_text"], ["review_id", "review_comment_message", "review_score"]].copy()
    texts = sub["review_comment_message"].astype(str).tolist()

    star_a = method_a_transformer(texts)
    compound_b = method_b_leia(texts)

    out = pd.DataFrame({
        "review_id": sub["review_id"].values,
        "star_a": star_a,
        "compound_b": compound_b,
        "review_score": sub["review_score"].values,
    })
    out["label_a"] = out["star_a"].map(stars_to_polarity)
    out["label_b"] = out["compound_b"].map(compound_to_polarity)
    out["label_truth"] = out["review_score"].map(stars_to_polarity)
    # Internal agreement (models vs each other) — inspect where they disagree, don't drop.
    out["ab_agree"] = out["label_a"] == out["label_b"]
    # External validity (each model vs the star ground truth) — the different job.
    out["a_valid"] = out["label_a"] == out["label_truth"]
    out["b_valid"] = out["label_b"] == out["label_truth"]
    return out


# --------------------------------------------------------------------------- #
# Chart 4 (Plotly) — regional bubble map
# --------------------------------------------------------------------------- #
def chart_regional_bubble(df: pd.DataFrame, min_orders=30, save_html=None, save_png=None):
    """City-level bubble map: size = order volume, color = late-delivery RATE.

    Color is the volume-normalized rate, never raw counts — otherwise the map just
    reproduces population. The min-order threshold drops small-n cities whose rates
    are noise; size doubles as a confidence channel (small city = small dot).
    """
    import plotly.express as px

    agg = (df.groupby("customer_city")
             .agg(n=("review_id", "size"),
                  late_rate=("is_late", "mean"),
                  avg_score=("review_score", "mean"),
                  lat=("customer_lat", "mean"),
                  lng=("customer_lng", "mean"),
                  state=("customer_state", "first"))
             .reset_index())
    agg = agg[(agg["n"] >= min_orders) & agg["lat"].notna()]

    fig = px.scatter_geo(
        agg, lat="lat", lon="lng", size="n", color="late_rate",
        color_continuous_scale="Reds", scope="south america", size_max=38,
        hover_name="customer_city",
        hover_data={"state": True, "late_rate": ":.1%", "avg_score": ":.2f",
                    "n": True, "lat": False, "lng": False},
        labels={"late_rate": "Late rate", "n": "Orders"},
        title=f"Where late deliveries concentrate (cities with ≥{min_orders} orders)")
    fig.update_geos(fitbounds="locations", showcountries=True)
    if save_html:
        fig.write_html(save_html)
    if save_png:
        fig.write_image(save_png, width=900, height=700, scale=2)
    return fig


# --------------------------------------------------------------------------- #
# Generative theming (the differentiator) — ALL provider code lives in get_themes
# --------------------------------------------------------------------------- #
# Fixed taxonomy so themes are comparable ACROSS batches (a freeform label set would
# drift — "late" vs "delayed" — and break both frequency ranking and the stability
# check). "other" is the escape hatch. Domain-derived for negative e-commerce reviews.
THEME_VOCAB = [
    "never_arrived",          # produto nunca chegou
    "late_delivery",          # chegou atrasado
    "wrong_or_missing_item",  # item errado / faltando
    "damaged_product",        # produto danificado
    "poor_quality",           # qualidade ruim / não funciona
    "not_as_described",       # diferente do anunciado
    "refund_or_billing",      # estorno / cobrança
    "other",
]

_THEME_PROMPT = """You label negative Brazilian e-commerce reviews (Portuguese) by their main complaint.
Assign each review EXACTLY ONE theme from this fixed list:
{vocab}

Also rate severity: "low", "medium", or "high".

Return ONLY a JSON array, one object per review, no prose, no markdown fences:
[{{"review_id": "<id>", "theme": "<one of the list>", "severity": "<low|medium|high>"}}]

Reviews:
{reviews}"""


def _parse_json_array(text: str):
    """Defensive parse: models sometimes wrap JSON in ```fences``` or add a stray line."""
    import json, re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # strip code fences, then grab the outermost [...] and retry
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end + 1])
        raise


def get_themes(review_batch, model="claude-haiku-4-5"):
    """LLM-theme one batch of negative reviews. Returns [{review_id, theme, severity}].

    review_batch: list of {"review_id": str, "text": str}. This is the ONLY function
    with provider-specific code — swap the body to change providers (see ADR-0001).
    Batching is forced by context window; the caller batches and concatenates.
    """
    import os, anthropic
    from dotenv import load_dotenv
    load_dotenv()

    reviews_block = "\n".join(
        f'- review_id={r["review_id"]}: {str(r["text"])[:400]}' for r in review_batch)
    prompt = _THEME_PROMPT.format(vocab="\n".join(f"- {t}" for t in THEME_VOCAB),
                                  reviews=reviews_block)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(2):  # one retry on malformed JSON
        msg = client.messages.create(model=model, max_tokens=2000,
                                     messages=[{"role": "user", "content": prompt}])
        try:
            parsed = _parse_json_array(msg.content[0].text)
            valid = {t for t in THEME_VOCAB}
            for row in parsed:  # clamp unknown labels to "other"
                if row.get("theme") not in valid:
                    row["theme"] = "other"
            return parsed
        except Exception:
            if attempt == 1:
                raise
    return []


def _sample_negatives(df, n_sample=400, seed=42):
    neg = df.loc[df["is_negative"] & df["has_text"], ["review_id", "review_comment_message"]]
    return neg.sample(min(n_sample, len(neg)), random_state=seed)


def build_themes(df, n_sample=400, batch_size=25, seed=42):
    """Theme a stratified-random sample of negative reviews, batched. Keyed to review_id.

    Sampling is a *design requirement* here, not a shortcut: the stability check needs
    same-size batches to compare. Cache via load_or_build('themes.pkl', ...).
    """
    sample = _sample_negatives(df, n_sample, seed)
    rows = []
    for i in range(0, len(sample), batch_size):
        chunk = sample.iloc[i:i + batch_size]
        batch = [{"review_id": r.review_id, "text": r.review_comment_message}
                 for r in chunk.itertuples()]
        rows.extend(get_themes(batch))
    return pd.DataFrame(rows)


def theme_stability(df, n_sample=400, batch_size=25, seed=42):
    """Run theming twice on the SAME sample; report how often the theme label agrees.

    Answers "how do you know it didn't hallucinate": a stable extractor lands the same
    review on the same theme across runs.
    """
    a = build_themes(df, n_sample, batch_size, seed).set_index("review_id")["theme"]
    b = build_themes(df, n_sample, batch_size, seed).set_index("review_id")["theme"]
    joined = a.to_frame("a").join(b.rename("b"), how="inner")
    return (joined["a"] == joined["b"]).mean()


# --------------------------------------------------------------------------- #
# Chart 3 (Plotly) — theme frequency, on-time/late split, hover → sample review
# --------------------------------------------------------------------------- #
def chart_theme_frequency(themes_df, gold_df, save_html=None, save_png=None):
    """Ranked theme frequency; each bar split by delivery outcome. One visual carries
    both the overall ranking and the late-delivery surge (avoids two co-equal charts)."""
    import plotly.express as px

    m = themes_df.merge(gold_df[["review_id", "is_late", "review_comment_message"]],
                        on="review_id", how="left")
    m["delivery_outcome"] = m["is_late"].map({False: "On-time", True: "Late"})
    order = m["theme"].value_counts().index.tolist()[::-1]  # most frequent on top
    # one representative review per theme for the hover
    example = (m.dropna(subset=["review_comment_message"])
                 .groupby("theme")["review_comment_message"].first().str[:120])
    plot = m.groupby(["theme", "delivery_outcome"]).size().reset_index(name="n")
    plot["example"] = plot["theme"].map(example)

    fig = px.bar(plot, x="n", y="theme", color="delivery_outcome", orientation="h",
                 category_orders={"theme": order, "delivery_outcome": OUTCOME_ORDER},
                 color_discrete_map=OUTCOME_PALETTE, custom_data=["example"],
                 title="Why negative reviews are negative — delivery themes dominate",
                 labels={"n": "Reviews", "theme": "", "delivery_outcome": "Delivery"})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x} reviews<br><i>%{customdata[0]}</i><extra></extra>")
    if save_html:
        fig.write_html(save_html)
    if save_png:
        fig.write_image(save_png, width=900, height=520, scale=2)
    return fig
