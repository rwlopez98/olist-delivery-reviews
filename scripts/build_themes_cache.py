"""Build the themes cache (data/cache/themes.pkl), run the stability check, render Chart 3."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib; matplotlib.use("Agg")
import olist

df = olist.add_features(olist.load_gold())

# Pass 1 — cached (reviewer default loads this; no API hit on a plain run-all).
themes = olist.load_or_build("themes.pkl", lambda: olist.build_themes(df))

# Pass 2 — same sample, fresh run — stability: does a review land on the same theme?
themes2 = olist.build_themes(df)
a = themes.set_index("review_id")["theme"]
b = themes2.set_index("review_id")["theme"]
both = a.to_frame("a").join(b.rename("b"), how="inner")
stability = (both["a"] == both["b"]).mean()

olist.chart_theme_frequency(themes, df,
                            save_html="reports/figures/chart3_themes.html",
                            save_png="reports/figures/chart3_themes.png")

freq = themes["theme"].value_counts()
print("THEMES_SUMMARY", json.dumps({
    "n": int(len(themes)),
    "stability_pct": round(stability * 100, 1),
    "freq": {k: int(v) for k, v in freq.items()},
}))
