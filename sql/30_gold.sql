-- GOLD — the analytical table: ONE ROW PER REVIEW.
--
-- Delivery facts are per-order, joined onto each review. Order-grain summaries (items,
-- payments) attach via their pre-aggregated views so the review grain stays intact.
-- Raw delivery DATES are carried through unchanged; the on-time/late flag and delivery-gap
-- are computed in pandas (feature engineering) — SQL owns the data layer, pandas owns the
-- analytical layer, and neither builds the same column twice.
--
-- The INNER join to orders_clean is what enforces the exclusion: reviews whose order has no
-- delivery date simply drop out. review text is retained for the sentiment + theming layers.

CREATE OR REPLACE VIEW gold_reviews AS
    SELECT
        r.review_id,                              -- grain key (unique after dedup)
        r.order_id,
        r.review_score,
        r.review_comment_title,
        r.review_comment_message,
        (r.review_comment_message IS NOT NULL)      AS has_text,

        o.order_purchase_timestamp,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,

        c.customer_unique_id,
        c.customer_state,
        c.customer_zip_code_prefix,
        c.customer_city,
        g.lat                                       AS customer_lat,
        g.lng                                       AS customer_lng,

        i.n_items,
        i.n_sellers,
        i.items_total,
        i.freight_total,
        p.payment_total,
        p.max_installments
    FROM reviews_dedup            r
    JOIN orders_clean            o USING (order_id)
    JOIN customers               c USING (customer_id)
    LEFT JOIN order_items_agg    i USING (order_id)
    LEFT JOIN order_payments_agg p USING (order_id)
    LEFT JOIN geolocation_centroid g
           ON c.customer_zip_code_prefix = g.zip_prefix;
