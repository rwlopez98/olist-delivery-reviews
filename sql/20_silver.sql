-- SILVER — clean, resolve grain, pre-aggregate. Every grain explosion is settled HERE
-- so the review grain in Gold is never contaminated.

-- Orders that were actually delivered. Lateness is undefined without a real delivery
-- date, so we exclude on the DATE being null — NOT on canceled status: the null-delivered
-- population (2,965) spans 7 statuses and does not equal the 625 canceled orders, and 8
-- orders marked 'delivered' carry no delivery date. The date null is the honest criterion.
CREATE OR REPLACE VIEW orders_clean AS
    SELECT
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_delivered_customer_date,
        order_estimated_delivery_date
    FROM orders
    WHERE order_delivered_customer_date IS NOT NULL;

-- One row per review. Raw order_reviews has 814 duplicate review_id rows (99,224 → 98,410
-- distinct), and some review_id map to MULTIPLE order_id. Dedup among DELIVERED orders only
-- (join orders_clean first) so a review is kept whenever it has any delivered order, then
-- keep the richest row per review_id. `order_id` is the final tiebreaker: without it the
-- ORDER BY has ties and ROW_NUMBER picks nondeterministically, which — combined with the
-- delivered-order filter — made the Gold row count wobble run-to-run.
CREATE OR REPLACE VIEW reviews_dedup AS
    SELECT * EXCLUDE (rn) FROM (
        SELECT rev.*,
            ROW_NUMBER() OVER (
                PARTITION BY rev.review_id
                ORDER BY (rev.review_comment_message IS NOT NULL) DESC,
                         rev.review_answer_timestamp DESC,
                         rev.order_id
            ) AS rn
        FROM order_reviews rev
        JOIN orders_clean USING (order_id)
    ) WHERE rn = 1;

-- Items → order grain. A multi-item order must contribute ONE delivery outcome, not one
-- per item, or every lateness average is silently weighted by basket size.
CREATE OR REPLACE VIEW order_items_agg AS
    SELECT
        order_id,
        COUNT(*)                        AS n_items,
        COUNT(DISTINCT seller_id)       AS n_sellers,
        SUM(price)                      AS items_total,
        SUM(freight_value)              AS freight_total
    FROM order_items
    GROUP BY order_id;

-- Payments → order grain (an order can have several payment rows).
CREATE OR REPLACE VIEW order_payments_agg AS
    SELECT
        order_id,
        SUM(payment_value)              AS payment_total,
        COUNT(*)                        AS n_payments,
        MAX(payment_installments)       AS max_installments
    FROM order_payments
    GROUP BY order_id;

-- Geolocation → one representative point per zip prefix. Raw table is ~1M rows with many
-- lat/lng per prefix; the centroid is all Chart 4 needs. Keeps this heavy table out of the
-- grain and touching the spine only through this collapsed view.
CREATE OR REPLACE VIEW geolocation_centroid AS
    SELECT
        geolocation_zip_code_prefix     AS zip_prefix,
        AVG(geolocation_lat)            AS lat,
        AVG(geolocation_lng)            AS lng,
        ANY_VALUE(geolocation_state)    AS geo_state
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix;
