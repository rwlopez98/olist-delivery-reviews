-- ZERVE GOLD — single SQL block. Bronze→Silver→Gold collapsed into CTEs over the
-- auto-loaded tables. Logic is identical to the tested offline 10/20/30_*.sql, but written
-- in portable SQL (no DuckDB-only `* EXCLUDE` / `ANY_VALUE`, explicit columns, ON joins) so
-- it runs on whatever engine Zerve's SQL blocks use.
--
-- ADJUST the table names below if Zerve names them differently than the CSV basenames.
-- If SQL blocks can chain (one block's output is a table the next can query), you may split
-- each CTE into its own block to show the medallion physically; the logic is unchanged.

WITH orders_clean AS (   -- SILVER: exclude undelivered (lateness undefined) — 2,965 rows drop
    SELECT order_id, customer_id, order_status,
           order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date
    FROM olist_orders_dataset
    WHERE order_delivered_customer_date IS NOT NULL
),
reviews_ranked AS (      -- SILVER: dedup reviews to one row per review_id among delivered orders
    SELECT rev.review_id, rev.order_id, rev.review_score,
           rev.review_comment_title, rev.review_comment_message,
           ROW_NUMBER() OVER (
               PARTITION BY rev.review_id
               ORDER BY CASE WHEN rev.review_comment_message IS NOT NULL THEN 0 ELSE 1 END,
                        rev.review_answer_timestamp DESC,
                        rev.order_id
           ) AS rn
    FROM olist_order_reviews_dataset rev
    JOIN orders_clean oc ON rev.order_id = oc.order_id
),
reviews_dedup AS (
    SELECT review_id, order_id, review_score, review_comment_title, review_comment_message
    FROM reviews_ranked
    WHERE rn = 1
),
order_items_agg AS (     -- SILVER: items → order grain (prevents lateness double-counting)
    SELECT order_id, COUNT(*) AS n_items, COUNT(DISTINCT seller_id) AS n_sellers,
           SUM(price) AS items_total, SUM(freight_value) AS freight_total
    FROM olist_order_items_dataset GROUP BY order_id
),
order_payments_agg AS (  -- SILVER: payments → order grain
    SELECT order_id, SUM(payment_value) AS payment_total, COUNT(*) AS n_payments,
           MAX(payment_installments) AS max_installments
    FROM olist_order_payments_dataset GROUP BY order_id
),
geolocation_centroid AS (-- SILVER: ~1M rows → one point per zip prefix (Chart 4 only)
    SELECT geolocation_zip_code_prefix AS zip_prefix,
           AVG(geolocation_lat) AS lat, AVG(geolocation_lng) AS lng,
           MAX(geolocation_state) AS geo_state
    FROM olist_geolocation_dataset GROUP BY geolocation_zip_code_prefix
)
SELECT                   -- GOLD: one row per review
    r.review_id, r.order_id, r.review_score, r.review_comment_title, r.review_comment_message,
    CASE WHEN r.review_comment_message IS NOT NULL THEN 1 ELSE 0 END AS has_text,
    o.order_purchase_timestamp, o.order_delivered_customer_date, o.order_estimated_delivery_date,
    c.customer_unique_id, c.customer_state, c.customer_zip_code_prefix, c.customer_city,
    g.lat AS customer_lat, g.lng AS customer_lng,
    i.n_items, i.n_sellers, i.items_total, i.freight_total,
    p.payment_total, p.max_installments
FROM reviews_dedup r
JOIN orders_clean            o ON r.order_id = o.order_id
JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
LEFT JOIN order_items_agg    i ON r.order_id = i.order_id
LEFT JOIN order_payments_agg p ON r.order_id = p.order_id
LEFT JOIN geolocation_centroid g ON c.customer_zip_code_prefix = g.zip_prefix;
