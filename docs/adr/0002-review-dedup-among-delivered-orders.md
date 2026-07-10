# Deduplicate reviews among delivered orders, with a deterministic tiebreaker

**Status:** accepted

`review_id` is not unique in raw `order_reviews` (99,224 rows → 98,410 distinct), and some `review_id` map to multiple `order_id`. The naive dedup (keep richest row per `review_id`, ordered by has-text then answer timestamp) had **ties with no deterministic tiebreaker**, so `ROW_NUMBER` chose arbitrarily per run — and because Gold then inner-joins delivered orders, a run that kept an *undelivered* order for a review silently dropped it. Result: the Gold row count wobbled (~95,60x) run-to-run and could discard valid delivered reviews.

Fix: in `reviews_dedup`, **join `orders_clean` (delivered orders only) before deduping**, and add **`order_id` as a final tiebreaker**. A review is now kept whenever it has any delivered order, the same row is chosen every run, and Gold is a stable **95,645** rows = distinct `review_id`.

## Consequences

- Gold grain is deterministic — required because sentiment/theme pickles are keyed to `review_id`; a wobbling grain would misalign the cache.
- The exclusion (delivered-only) is effectively applied inside the dedup as well as at the Gold join; the Gold join to `orders_clean` remains (it supplies the delivery dates) and is now guaranteed to match.
- Do **not** "simplify" the dedup by removing the `orders_clean` join or the `order_id` tiebreaker — either reintroduces the nondeterminism.
