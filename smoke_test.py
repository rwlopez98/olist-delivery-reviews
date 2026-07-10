"""
§2 environment smoke-test — probe the walls before building on them.

Run offline first: `python smoke_test.py`. Four checks:
  1. DuckDB     — standard SQL over a dataframe (the offline SQL seam)
  2. LeIA       — Portuguese-VADER sentiment (Method B); the risky install
  3. HF pt-BR   — nlptown transformer sentiment (Method A), 1-5 star output
  4. Anthropic  — SDK import + live call (skipped until ANTHROPIC_API_KEY is set)

Checks 1-3 must pass offline. Check 4 is expected to skip offline and is the
only wall that can't be verified until the key exists / we're in Zerve.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env (gitignored)

PT_SAMPLES = [
    "O produto chegou muito atrasado, péssimo!",        # late + terrible  -> negative
    "Adorei, entrega super rápida e produto ótimo",     # fast + great     -> positive
    "Chegou no prazo, tudo certo",                       # on-time          -> neutral/pos
]


def check_duckdb():
    import duckdb, pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    got = duckdb.sql("select sum(a) as s from df").df().iloc[0, 0]
    assert got == 6, got
    return f"duckdb {duckdb.__version__} — SQL over dataframe OK"


def check_leia():
    from LeIA import SentimentIntensityAnalyzer  # pip package: leia-br
    s = SentimentIntensityAnalyzer()
    scores = [s.polarity_scores(t)["compound"] for t in PT_SAMPLES]
    assert scores[0] < 0 < scores[1], scores  # late<0<fast
    return "LeIA (leia-br) — compound " + ", ".join(f"{x:+.2f}" for x in scores)


def check_hf_transformer():
    from transformers import pipeline
    clf = pipeline("sentiment-analysis",
                   model="nlptown/bert-base-multilingual-uncased-sentiment")
    labels = [clf(t)[0]["label"] for t in PT_SAMPLES]
    assert labels[0].startswith("1"), labels  # late+terrible -> 1 star
    return "nlptown pt-BR transformer — labels " + ", ".join(labels)


def check_anthropic():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "SKIP — anthropic SDK importable, no ANTHROPIC_API_KEY set (expected offline)"
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-haiku-4-5", max_tokens=16,
        messages=[{"role": "user", "content": "reply with the single word: ok"}],
    )
    return f"anthropic live call OK — {msg.content[0].text.strip()!r}"


if __name__ == "__main__":
    for name, fn in [("DuckDB", check_duckdb), ("LeIA", check_leia),
                     ("HF pt-BR", check_hf_transformer), ("Anthropic", check_anthropic)]:
        try:
            print(f"[PASS] {name}: {fn()}")
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
