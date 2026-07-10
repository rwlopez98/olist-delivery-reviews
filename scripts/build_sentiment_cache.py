"""Build the full sentiment cache (data/cache/sentiment.pkl) over all text reviews.
Run offline once; the pickle then crosses into Zerve. Prints a validation summary.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import olist

df = olist.add_features(olist.load_gold())
sent = olist.load_or_build("sentiment.pkl", lambda: olist.build_sentiment(df))

summary = {
    "n": int(len(sent)),
    "a_valid_pct": round(sent.a_valid.mean() * 100, 1),
    "b_valid_pct": round(sent.b_valid.mean() * 100, 1),
    "ab_agree_pct": round(sent.ab_agree.mean() * 100, 1),
    "disagree_n": int((~sent.ab_agree).sum()),
}
print("SENTIMENT_SUMMARY", json.dumps(summary))
