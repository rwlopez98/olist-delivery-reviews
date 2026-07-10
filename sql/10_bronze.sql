-- BRONZE — acknowledge the raw tables, scope to the spine.
--
-- The Olist dataset auto-loads NINE tables. This analysis uses the delivery→reviews
-- spine only:
--   in-spine : orders, order_reviews, order_items, order_payments, customers
--              (+ geolocation for the Chart-4 regional map)
--   out-of-spine (named, not used): sellers, products,
--              product_category_name_translation — product/seller attributes don't
--              serve the delivery→review question; seller stays a supporting cut at most.
--
-- PORTABILITY: this file is the ONLY place bound to the data source. `${DATA_PREFIX}` is
-- substituted at load time from a single config variable (olist.DATA_PREFIX), so relocating
-- the data — or embedding into Zerve's folder structure — is a one-line change. In Zerve you
-- either point DATA_PREFIX at the CSVs or replace these views with the auto-loaded tables.

CREATE OR REPLACE VIEW orders AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_orders_dataset.csv');

CREATE OR REPLACE VIEW order_reviews AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_order_reviews_dataset.csv');

CREATE OR REPLACE VIEW order_items AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_order_items_dataset.csv');

CREATE OR REPLACE VIEW order_payments AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_order_payments_dataset.csv');

CREATE OR REPLACE VIEW customers AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_customers_dataset.csv');

CREATE OR REPLACE VIEW geolocation AS
    SELECT * FROM read_csv_auto('${DATA_PREFIX}/olist_geolocation_dataset.csv');
