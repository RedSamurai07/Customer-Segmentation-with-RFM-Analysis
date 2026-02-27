-- ============================================================
-- CUSTOMER SEGMENTATION — COMPLETE SQL ANALYSIS
-- UCI Online Retail II Dataset
-- Platform: BigQuery / PostgreSQL
-- ============================================================
-- Covers: Revenue Analysis, Cohort Retention, Product Analysis,
-- Customer Behavior, Advanced Window Functions, Return Analysis
-- ============================================================

-- NOTE: Replace `your_project.your_dataset.online_retail`
-- with your actual table name in BigQuery, OR use the table
-- name directly if running in PostgreSQL.

-- Assumed table schema:
-- Invoice        STRING
-- StockCode      STRING
-- Description    STRING
-- Quantity       INT64
-- InvoiceDate    TIMESTAMP
-- Price          FLOAT64
-- CustomerID     FLOAT64
-- Country        STRING


-- ─────────────────────────────────────────────────────────────
-- QUERY 1: COMPREHENSIVE DATA QUALITY AUDIT
-- Categorize every row into 5 transaction types
-- ─────────────────────────────────────────────────────────────

SELECT
    CASE
        WHEN STARTS_WITH(Invoice, 'C')          THEN 'Cancellation'
        WHEN Price < 0                           THEN 'Negative_Price_Adjustment'
        WHEN Price = 0 AND Quantity > 0          THEN 'Free_Sample_Promotional'
        WHEN Quantity < 0 AND NOT STARTS_WITH(Invoice, 'C')
                                                 THEN 'Return_Without_C_Flag'
        ELSE                                          'Valid_Purchase'
    END                                          AS Row_Type,
    COUNT(*)                                     AS Row_Count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2)
                                                 AS Row_Pct,
    ROUND(SUM(Quantity * Price), 2)              AS Total_Revenue,
    ROUND(AVG(Price), 2)                         AS Avg_Price,
    ROUND(AVG(Quantity), 2)                      AS Avg_Quantity
FROM
    `your_project.your_dataset.online_retail`
GROUP BY
    Row_Type
ORDER BY
    Row_Count DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 2: MISSING CUSTOMER ID PATTERN ANALYSIS
-- Compare purchase behavior between missing vs present IDs
-- ─────────────────────────────────────────────────────────────

SELECT
    CASE WHEN CustomerID IS NULL THEN 'Missing_ID' ELSE 'Has_ID' END
                                                 AS ID_Status,
    COUNT(*)                                     AS Transaction_Count,
    ROUND(AVG(Quantity * Price), 2)              AS Avg_Order_Value,
    ROUND(AVG(Quantity), 2)                      AS Avg_Quantity,
    COUNT(DISTINCT Country)                      AS Unique_Countries,
    ROUND(AVG(EXTRACT(HOUR FROM InvoiceDate)), 1)
                                                 AS Avg_Purchase_Hour,
    COUNT(DISTINCT Country)                      AS Country_Spread
FROM
    `your_project.your_dataset.online_retail`
GROUP BY
    ID_Status;

-- Top countries with missing Customer IDs
SELECT
    Country,
    COUNT(*)                                     AS Missing_ID_Transactions,
    ROUND(AVG(Quantity * Price), 2)              AS Avg_Revenue
FROM
    `your_project.your_dataset.online_retail`
WHERE
    CustomerID IS NULL
GROUP BY
    Country
ORDER BY
    Missing_ID_Transactions DESC
LIMIT 15;


-- ─────────────────────────────────────────────────────────────
-- QUERY 3: FULL RFM CALCULATION IN SQL
-- Using NTILE window function for 1-5 scoring
-- ─────────────────────────────────────────────────────────────

WITH clean_data AS (
    -- Step 1: Filter to valid purchases only
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        Invoice,
        InvoiceDate,
        Quantity * Price                         AS LineRevenue
    FROM
        `your_project.your_dataset.online_retail`
    WHERE
        CustomerID IS NOT NULL
        AND NOT STARTS_WITH(Invoice, 'C')
        AND Quantity > 0
        AND Price > 0
),

rfm_base AS (
    -- Step 2: Calculate raw RFM metrics
    SELECT
        CustomerID,
        DATE_DIFF(
            DATE '2011-12-10',                   -- Reference date (day after last transaction)
            DATE(MAX(InvoiceDate)),
            DAY
        )                                        AS Recency_Days,
        COUNT(DISTINCT Invoice)                  AS Frequency,
        ROUND(SUM(LineRevenue), 2)               AS Monetary
    FROM
        clean_data
    GROUP BY
        CustomerID
),

rfm_scored AS (
    -- Step 3: Apply NTILE scoring (1-5 per dimension)
    SELECT
        CustomerID,
        Recency_Days,
        Frequency,
        Monetary,
        -- Recency: lower days = higher score (reversed)
        6 - NTILE(5) OVER (ORDER BY Recency_Days ASC)
                                                 AS R_Score,
        NTILE(5) OVER (ORDER BY Frequency ASC)  AS F_Score,
        NTILE(5) OVER (ORDER BY Monetary ASC)   AS M_Score
    FROM
        rfm_base
),

rfm_combined AS (
    -- Step 4: Create composite score and segment label
    SELECT
        CustomerID,
        Recency_Days,
        Frequency,
        Monetary,
        R_Score,
        F_Score,
        M_Score,
        (R_Score + F_Score + M_Score)            AS RFM_Score,
        -- Weighted score: R=0.3, F=0.3, M=0.4
        ROUND(R_Score * 0.3 + F_Score * 0.3 + M_Score * 0.4, 2)
                                                 AS Weighted_RFM,
        CASE
            WHEN R_Score >= 4 AND F_Score >= 4 AND M_Score >= 4
                                                 THEN 'Champions'
            WHEN R_Score >= 3 AND F_Score >= 3   THEN 'Loyal_Customers'
            WHEN R_Score >= 4 AND F_Score <= 2   THEN 'Recent_Customers'
            WHEN R_Score >= 3 AND M_Score >= 3   THEN 'Potential_Loyalists'
            WHEN R_Score = 5                     THEN 'New_Customers'
            WHEN R_Score <= 2 AND F_Score >= 3   THEN 'At_Risk'
            WHEN R_Score <= 2 AND F_Score <= 2 AND M_Score >= 3
                                                 THEN 'Cannot_Lose_Them'
            WHEN R_Score <= 2 AND F_Score <= 2   THEN 'Lost'
            ELSE                                      'Hibernating'
        END                                      AS Segment
    FROM
        rfm_scored
)

SELECT *
FROM rfm_combined
ORDER BY Monetary DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 4: PARETO ANALYSIS — CUMULATIVE REVENUE
-- Identify the % of customers driving 80% of revenue
-- ─────────────────────────────────────────────────────────────

WITH customer_revenue AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue
    FROM
        `your_project.your_dataset.online_retail`
    WHERE
        CustomerID IS NOT NULL
        AND NOT STARTS_WITH(Invoice, 'C')
        AND Quantity > 0 AND Price > 0
    GROUP BY
        CustomerID
),

ranked AS (
    SELECT
        CustomerID,
        Total_Revenue,
        ROW_NUMBER() OVER (ORDER BY Total_Revenue DESC)
                                                 AS Revenue_Rank,
        COUNT(*) OVER ()                         AS Total_Customers,
        SUM(Total_Revenue) OVER (
            ORDER BY Total_Revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                        AS Cumulative_Revenue,
        SUM(Total_Revenue) OVER ()               AS Grand_Total_Revenue
    FROM
        customer_revenue
)

SELECT
    CustomerID,
    Total_Revenue,
    Revenue_Rank,
    ROUND(Revenue_Rank * 100.0 / Total_Customers, 2)
                                                 AS Customer_Percentile,
    ROUND(Cumulative_Revenue * 100.0 / Grand_Total_Revenue, 2)
                                                 AS Cumulative_Revenue_Pct,
    CASE
        WHEN Cumulative_Revenue * 100.0 / Grand_Total_Revenue <= 80
                                                 THEN 'Top_Revenue_Drivers'
        ELSE                                          'Remaining_Customers'
    END                                          AS Pareto_Group
FROM
    ranked
ORDER BY
    Revenue_Rank;

-- Summary: What customer percentile generates 80% of revenue?
WITH customer_revenue AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID
),
ranked AS (
    SELECT
        CustomerID, Total_Revenue,
        ROW_NUMBER() OVER (ORDER BY Total_Revenue DESC) AS Rnk,
        COUNT(*) OVER () AS Total_Customers,
        SUM(Total_Revenue) OVER (
            ORDER BY Total_Revenue DESC
            ROWS UNBOUNDED PRECEDING
        ) AS Cumul_Rev,
        SUM(Total_Revenue) OVER () AS Grand_Total
    FROM customer_revenue
)
SELECT
    MAX(CASE WHEN Cumul_Rev * 100.0 / Grand_Total <= 80
             THEN ROUND(Rnk * 100.0 / Total_Customers, 1) END)
                                                 AS Pct_Customers_For_80Pct_Revenue,
    MAX(CASE WHEN Cumul_Rev * 100.0 / Grand_Total <= 80 THEN Rnk END)
                                                 AS Customer_Count_For_80Pct_Revenue
FROM ranked;


-- ─────────────────────────────────────────────────────────────
-- QUERY 5: COHORT RETENTION TABLE
-- Full month-by-month retention matrix
-- ─────────────────────────────────────────────────────────────

WITH clean AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        DATE_TRUNC(DATE(InvoiceDate), MONTH)     AS Purchase_Month
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
),

first_purchase AS (
    SELECT
        CustomerID,
        MIN(Purchase_Month)                      AS Cohort_Month
    FROM clean
    GROUP BY CustomerID
),

cohort_data AS (
    SELECT
        c.CustomerID,
        f.Cohort_Month,
        c.Purchase_Month,
        DATE_DIFF(c.Purchase_Month, f.Cohort_Month, MONTH)
                                                 AS Months_Since_First
    FROM clean c
    JOIN first_purchase f USING (CustomerID)
),

cohort_sizes AS (
    SELECT
        Cohort_Month,
        COUNT(DISTINCT CustomerID)               AS Cohort_Size
    FROM first_purchase
    GROUP BY Cohort_Month
),

retention_raw AS (
    SELECT
        Cohort_Month,
        Months_Since_First,
        COUNT(DISTINCT CustomerID)               AS Active_Customers
    FROM cohort_data
    GROUP BY Cohort_Month, Months_Since_First
)

SELECT
    r.Cohort_Month,
    cs.Cohort_Size,
    r.Months_Since_First,
    r.Active_Customers,
    ROUND(r.Active_Customers * 100.0 / cs.Cohort_Size, 1)
                                                 AS Retention_Rate_Pct
FROM retention_raw r
JOIN cohort_sizes cs USING (Cohort_Month)
WHERE r.Months_Since_First <= 11               -- Show up to 12 months
ORDER BY r.Cohort_Month, r.Months_Since_First;


-- ─────────────────────────────────────────────────────────────
-- QUERY 6: CHURNED CUSTOMERS IDENTIFICATION
-- Active in first half, silent in second half
-- ─────────────────────────────────────────────────────────────

WITH customer_halves AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN 1 ELSE 0 END)              AS Active_H1,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-06-01' AND '2010-12-31'
                 THEN 1 ELSE 0 END)              AS Active_H2,
        SUM(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN Quantity * Price ELSE 0 END)
                                                 AS H1_Revenue,
        COUNT(DISTINCT CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                            THEN Invoice END)    AS H1_Frequency,
        DATE_DIFF(
            DATE '2010-05-31',
            DATE(MIN(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                          THEN InvoiceDate END)),
            DAY
        )                                        AS H1_Recency_At_End
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID
)

SELECT
    CustomerID,
    H1_Revenue,
    H1_Frequency,
    H1_Recency_At_End,
    Active_H1,
    Active_H2,
    CASE WHEN Active_H1 = 1 AND Active_H2 = 0 THEN 'Churned'
         WHEN Active_H1 = 1 AND Active_H2 = 1 THEN 'Retained'
         WHEN Active_H1 = 0 AND Active_H2 = 1 THEN 'Reactivated'
         ELSE 'Inactive_Both'
    END                                          AS Customer_Status
FROM customer_halves
WHERE Active_H1 = 1
ORDER BY H1_Revenue DESC;

-- Summary of churn metrics
WITH -- (paste CTE above here)
customer_halves AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN 1 ELSE 0 END)              AS Active_H1,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-06-01' AND '2010-12-31'
                 THEN 1 ELSE 0 END)              AS Active_H2,
        SUM(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN Quantity * Price ELSE 0 END) AS H1_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID
)
SELECT
    ROUND(SUM(CASE WHEN Active_H1=1 AND Active_H2=0 THEN 1 ELSE 0 END) * 100.0
          / NULLIF(SUM(Active_H1), 0), 2)        AS Churn_Rate_Pct,
    ROUND(AVG(CASE WHEN Active_H1=1 AND Active_H2=0 THEN H1_Revenue END), 2)
                                                 AS Avg_Revenue_Of_Churned,
    ROUND(AVG(CASE WHEN Active_H1=1 AND Active_H2=1 THEN H1_Revenue END), 2)
                                                 AS Avg_Revenue_Of_Retained
FROM customer_halves;


-- ─────────────────────────────────────────────────────────────
-- QUERY 7: INTER-PURCHASE INTERVAL ANALYSIS (Recursive CTE)
-- Days between each consecutive purchase per customer
-- ─────────────────────────────────────────────────────────────

WITH customer_purchases AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        DATE(InvoiceDate)                        AS Purchase_Date,
        ROW_NUMBER() OVER (
            PARTITION BY CustomerID
            ORDER BY DATE(InvoiceDate)
        )                                        AS Purchase_Number
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID, DATE(InvoiceDate)      -- one row per customer-day
),

with_lag AS (
    SELECT
        CustomerID,
        Purchase_Date,
        Purchase_Number,
        LAG(Purchase_Date) OVER (
            PARTITION BY CustomerID
            ORDER BY Purchase_Date
        )                                        AS Prev_Purchase_Date
    FROM customer_purchases
),

intervals AS (
    SELECT
        CustomerID,
        Purchase_Date,
        Purchase_Number,
        DATE_DIFF(Purchase_Date, Prev_Purchase_Date, DAY)
                                                 AS Days_Since_Last_Purchase
    FROM with_lag
    WHERE Prev_Purchase_Date IS NOT NULL
)

SELECT
    CustomerID,
    ROUND(AVG(Days_Since_Last_Purchase), 1)     AS Avg_Interval_Days,
    MIN(Days_Since_Last_Purchase)               AS Min_Interval_Days,
    MAX(Days_Since_Last_Purchase)               AS Max_Interval_Days,
    ROUND(STDDEV(Days_Since_Last_Purchase), 1)  AS Std_Interval_Days,
    COUNT(*)                                    AS Num_Intervals,
    -- Is interval trend increasing? Compare first half vs second half of purchases
    ROUND(AVG(CASE WHEN Purchase_Number > (SELECT MAX(Purchase_Number)/2
                                           FROM intervals i2
                                           WHERE i2.CustomerID = intervals.CustomerID)
                   THEN Days_Since_Last_Purchase END)
          - AVG(CASE WHEN Purchase_Number <= (SELECT MAX(Purchase_Number)/2
                                              FROM intervals i2
                                              WHERE i2.CustomerID = intervals.CustomerID)
                     THEN Days_Since_Last_Purchase END), 1)
                                                 AS Interval_Trend_Days  -- Positive = slowing down
FROM intervals
GROUP BY CustomerID
HAVING COUNT(*) >= 2
ORDER BY Interval_Trend_Days DESC;             -- Top = customers slowing down most


-- ─────────────────────────────────────────────────────────────
-- QUERY 8: PRODUCT CO-OCCURRENCE (SQL MARKET BASKET)
-- Association rules and lift calculation in SQL
-- ─────────────────────────────────────────────────────────────

WITH valid_transactions AS (
    SELECT
        Invoice,
        TRIM(Description)                        AS Product
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND Description IS NOT NULL
      AND Country = 'United Kingdom'             -- Focus on UK for volume
),

product_freq AS (
    SELECT
        Product,
        COUNT(DISTINCT Invoice)                  AS Product_Transactions
    FROM valid_transactions
    GROUP BY Product
),

total_transactions AS (
    SELECT COUNT(DISTINCT Invoice)               AS N
    FROM valid_transactions
),

co_occurrence AS (
    SELECT
        a.Product                                AS Product_A,
        b.Product                                AS Product_B,
        COUNT(DISTINCT a.Invoice)                AS Co_Occurrence_Count
    FROM valid_transactions a
    JOIN valid_transactions b
        ON  a.Invoice = b.Invoice
        AND a.Product < b.Product               -- Avoid duplicates (A,B) vs (B,A)
    GROUP BY a.Product, b.Product
    HAVING COUNT(DISTINCT a.Invoice) >= 20      -- Minimum support threshold
)

SELECT
    c.Product_A,
    c.Product_B,
    c.Co_Occurrence_Count,
    pa.Product_Transactions                      AS Freq_A,
    pb.Product_Transactions                      AS Freq_B,
    t.N                                          AS Total_Transactions,
    -- Support = P(A and B)
    ROUND(c.Co_Occurrence_Count * 1.0 / t.N, 4) AS Support,
    -- Confidence = P(B|A) = P(A and B) / P(A)
    ROUND(c.Co_Occurrence_Count * 1.0 / pa.Product_Transactions, 4)
                                                 AS Confidence,
    -- Lift = P(A and B) / (P(A) * P(B))
    ROUND(
        (c.Co_Occurrence_Count * 1.0 / t.N)
        / ((pa.Product_Transactions * 1.0 / t.N) *
           (pb.Product_Transactions * 1.0 / t.N)), 3
    )                                            AS Lift
FROM co_occurrence c
JOIN product_freq pa ON c.Product_A = pa.Product
JOIN product_freq pb ON c.Product_B = pb.Product
CROSS JOIN total_transactions t
ORDER BY Lift DESC
LIMIT 50;


-- ─────────────────────────────────────────────────────────────
-- QUERY 9: REVENUE PER UNIQUE CUSTOMER BY PRODUCT
-- Premium products with highest individual customer value
-- ─────────────────────────────────────────────────────────────

SELECT
    TRIM(Description)                            AS Product,
    COUNT(DISTINCT CustomerID)                   AS Unique_Customers,
    ROUND(SUM(Quantity * Price), 2)              AS Total_Revenue,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT CustomerID), 0), 2)
                                                 AS Revenue_Per_Customer,
    ROUND(AVG(Quantity * Price), 2)              AS Avg_Transaction_Value,
    COUNT(DISTINCT Invoice)                      AS Total_Purchases
FROM `your_project.your_dataset.online_retail`
WHERE CustomerID IS NOT NULL
  AND NOT STARTS_WITH(Invoice, 'C')
  AND Quantity > 0 AND Price > 0
  AND Description IS NOT NULL
GROUP BY TRIM(Description)
HAVING COUNT(DISTINCT CustomerID) >= 10         -- Exclude one-off items
ORDER BY Revenue_Per_Customer DESC
LIMIT 30;


-- ─────────────────────────────────────────────────────────────
-- QUERY 10: DECLINING PRODUCT SALES VELOCITY
-- Products with >30% sales drop Q4 vs Q1
-- ─────────────────────────────────────────────────────────────

WITH quarterly_sales AS (
    SELECT
        TRIM(Description)                        AS Product,
        SUM(CASE WHEN EXTRACT(MONTH FROM InvoiceDate) BETWEEN 1 AND 3
                 THEN Quantity ELSE 0 END)       AS Q1_Units,
        SUM(CASE WHEN EXTRACT(MONTH FROM InvoiceDate) BETWEEN 4 AND 6
                 THEN Quantity ELSE 0 END)       AS Q2_Units,
        SUM(CASE WHEN EXTRACT(MONTH FROM InvoiceDate) BETWEEN 7 AND 9
                 THEN Quantity ELSE 0 END)       AS Q3_Units,
        SUM(CASE WHEN EXTRACT(MONTH FROM InvoiceDate) BETWEEN 10 AND 12
                 THEN Quantity ELSE 0 END)       AS Q4_Units,
        SUM(Quantity * Price)                    AS Total_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND EXTRACT(YEAR FROM InvoiceDate) = 2010
      AND Description IS NOT NULL
    GROUP BY TRIM(Description)
    HAVING SUM(CASE WHEN EXTRACT(MONTH FROM InvoiceDate) BETWEEN 1 AND 3
                    THEN Quantity ELSE 0 END) >= 50  -- Minimum Q1 volume
)

SELECT
    Product,
    Q1_Units,
    Q2_Units,
    Q3_Units,
    Q4_Units,
    ROUND(Total_Revenue, 2)                      AS Total_Revenue,
    ROUND((Q4_Units - Q1_Units) * 100.0
          / NULLIF(Q1_Units, 0), 1)              AS Q4_vs_Q1_Change_Pct,
    CASE
        WHEN (Q4_Units - Q1_Units) * 100.0 / NULLIF(Q1_Units, 0) < -30
                                                 THEN 'DISCONTINUE_CANDIDATE'
        WHEN (Q4_Units - Q1_Units) * 100.0 / NULLIF(Q1_Units, 0) < 0
                                                 THEN 'DECLINING'
        WHEN (Q4_Units - Q1_Units) * 100.0 / NULLIF(Q1_Units, 0) > 20
                                                 THEN 'GROWING'
        ELSE                                          'STABLE'
    END                                          AS Trend_Flag
FROM quarterly_sales
ORDER BY Q4_vs_Q1_Change_Pct ASC
LIMIT 40;


-- ─────────────────────────────────────────────────────────────
-- QUERY 11: CUSTOMER COUNTRY PERCENTILE ANALYSIS
-- Top 10% within country but not globally
-- ─────────────────────────────────────────────────────────────

WITH customer_revenue AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        Country,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID, Country
),

with_percentiles AS (
    SELECT
        CustomerID,
        Country,
        Total_Revenue,
        PERCENT_RANK() OVER (
            PARTITION BY Country
            ORDER BY Total_Revenue ASC
        )                                        AS Country_Percentile,
        PERCENT_RANK() OVER (
            ORDER BY Total_Revenue ASC
        )                                        AS Global_Percentile
    FROM customer_revenue
)

SELECT
    CustomerID,
    Country,
    Total_Revenue,
    ROUND(Country_Percentile * 100, 1)           AS Country_Percentile_Pct,
    ROUND(Global_Percentile * 100, 1)            AS Global_Percentile_Pct,
    CASE
        WHEN Country_Percentile >= 0.90 AND Global_Percentile < 0.90
                                                 THEN 'Local_Champion_Underserved'
        WHEN Global_Percentile >= 0.90           THEN 'Global_Champion'
        WHEN Country_Percentile >= 0.90          THEN 'Local_Champion'
        ELSE                                          'Standard'
    END                                          AS Customer_Tier
FROM with_percentiles
WHERE Country_Percentile >= 0.90
ORDER BY Country, Country_Percentile DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 12: ANOMALOUS ORDER DETECTION
-- Orders >3 std deviations above customer average
-- ─────────────────────────────────────────────────────────────

WITH invoice_totals AS (
    SELECT
        Invoice,
        CAST(CustomerID AS INT64)                AS CustomerID,
        DATE(InvoiceDate)                        AS Invoice_Date,
        ROUND(SUM(Quantity * Price), 2)          AS Invoice_Total
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY Invoice, CustomerID, DATE(InvoiceDate)
),

customer_stats AS (
    SELECT
        CustomerID,
        AVG(Invoice_Total)                       AS Avg_Invoice,
        STDDEV(Invoice_Total)                    AS Std_Invoice,
        COUNT(*)                                 AS Total_Invoices
    FROM invoice_totals
    GROUP BY CustomerID
    HAVING COUNT(*) >= 3                         -- Need at least 3 orders for meaningful std
)

SELECT
    i.Invoice,
    i.CustomerID,
    i.Invoice_Date,
    i.Invoice_Total,
    ROUND(cs.Avg_Invoice, 2)                     AS Customer_Avg_Invoice,
    ROUND(cs.Std_Invoice, 2)                     AS Customer_Std_Invoice,
    ROUND((i.Invoice_Total - cs.Avg_Invoice)
          / NULLIF(cs.Std_Invoice, 0), 2)        AS Z_Score,
    CASE
        WHEN (i.Invoice_Total - cs.Avg_Invoice)
             / NULLIF(cs.Std_Invoice, 0) > 3    THEN 'ANOMALOUS_HIGH'
        WHEN (i.Invoice_Total - cs.Avg_Invoice)
             / NULLIF(cs.Std_Invoice, 0) < -3   THEN 'ANOMALOUS_LOW'
        ELSE                                          'NORMAL'
    END                                          AS Order_Classification
FROM invoice_totals i
JOIN customer_stats cs USING (CustomerID)
WHERE ABS((i.Invoice_Total - cs.Avg_Invoice)
          / NULLIF(cs.Std_Invoice, 0)) > 3
ORDER BY Z_Score DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 13: DAY-HOUR REVENUE HEATMAP (Fully in SQL)
-- Average revenue per transaction by day × hour
-- ─────────────────────────────────────────────────────────────

SELECT
    FORMAT_DATE('%A', DATE(InvoiceDate))         AS Day_Of_Week,
    EXTRACT(HOUR FROM InvoiceDate)               AS Hour_Of_Day,
    COUNT(DISTINCT Invoice)                      AS Num_Transactions,
    ROUND(SUM(Quantity * Price), 2)              AS Total_Revenue,
    ROUND(AVG(Quantity * Price), 2)              AS Avg_Revenue_Per_Line,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT Invoice), 0), 2)
                                                 AS Avg_Revenue_Per_Invoice,
    -- Rank within each day
    RANK() OVER (
        PARTITION BY FORMAT_DATE('%A', DATE(InvoiceDate))
        ORDER BY AVG(Quantity * Price) DESC
    )                                            AS Hour_Rank_Within_Day
FROM `your_project.your_dataset.online_retail`
WHERE CustomerID IS NOT NULL
  AND NOT STARTS_WITH(Invoice, 'C')
  AND Quantity > 0 AND Price > 0
GROUP BY Day_Of_Week, Hour_Of_Day
ORDER BY
    CASE Day_Of_Week
        WHEN 'Monday'    THEN 1
        WHEN 'Tuesday'   THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday'  THEN 4
        WHEN 'Friday'    THEN 5
        WHEN 'Saturday'  THEN 6
        WHEN 'Sunday'    THEN 7
    END,
    Hour_Of_Day;


-- ─────────────────────────────────────────────────────────────
-- QUERY 14: RETURN MATCHING & NET REVENUE PER CUSTOMER
-- Match cancellations to original purchases
-- ─────────────────────────────────────────────────────────────

WITH purchases AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        StockCode,
        Description,
        SUM(Quantity)                            AS Units_Purchased,
        SUM(Quantity * Price)                    AS Gross_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID, StockCode, Description
),

returns AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        StockCode,
        SUM(ABS(Quantity))                       AS Units_Returned,
        SUM(ABS(Quantity * Price))               AS Return_Value
    FROM `your_project.your_dataset.online_retail`
    WHERE STARTS_WITH(Invoice, 'C')
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID, StockCode
),

net_revenue AS (
    SELECT
        p.CustomerID,
        SUM(p.Gross_Revenue)                     AS Gross_Revenue,
        COALESCE(SUM(r.Return_Value), 0)         AS Total_Returns,
        SUM(p.Gross_Revenue) - COALESCE(SUM(r.Return_Value), 0)
                                                 AS Net_Revenue,
        COALESCE(SUM(r.Units_Returned), 0)
            / NULLIF(SUM(p.Units_Purchased), 0) AS Return_Rate
    FROM purchases p
    LEFT JOIN returns r
        ON  p.CustomerID = r.CustomerID
        AND p.StockCode  = r.StockCode
    GROUP BY p.CustomerID
)

SELECT
    CustomerID,
    ROUND(Gross_Revenue, 2)                      AS Gross_Revenue,
    ROUND(Total_Returns, 2)                      AS Total_Returns,
    ROUND(Net_Revenue, 2)                        AS Net_Revenue,
    ROUND(Return_Rate * 100, 2)                  AS Return_Rate_Pct,
    CASE
        WHEN Net_Revenue < 0                     THEN 'UNPROFITABLE'
        WHEN Return_Rate > 0.3                   THEN 'HIGH_RETURN_RISK'
        WHEN Return_Rate > 0.1                   THEN 'MODERATE_RETURN'
        ELSE                                          'LOW_RETURN'
    END                                          AS Return_Risk_Flag
FROM net_revenue
ORDER BY Return_Rate DESC;

-- Threshold: at what return rate does a customer become unprofitable?
WITH net_revenue AS (
    -- (paste CTE from above)
    SELECT
        CAST(CustomerID AS INT64) AS CustomerID,
        SUM(CASE WHEN NOT STARTS_WITH(Invoice,'C') THEN Quantity*Price ELSE 0 END) AS Gross,
        SUM(CASE WHEN STARTS_WITH(Invoice,'C') THEN ABS(Quantity*Price) ELSE 0 END) AS Returns
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
)
SELECT
    ROUND(Returns / NULLIF(Gross, 0) * 100)      AS Return_Rate_Rounded_Pct,
    COUNT(*)                                      AS Customer_Count,
    ROUND(AVG(Gross - Returns), 2)                AS Avg_Net_Revenue,
    ROUND(SUM(CASE WHEN Gross - Returns < 0 THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 1)                          AS Pct_Unprofitable
FROM net_revenue
WHERE Gross > 0
GROUP BY ROUND(Returns / NULLIF(Gross, 0) * 100)
ORDER BY Return_Rate_Rounded_Pct;


-- ─────────────────────────────────────────────────────────────
-- QUERY 15: RETURN RATE BY PRODUCT
-- Top returned products and profitability impact
-- ─────────────────────────────────────────────────────────────

WITH product_sold AS (
    SELECT
        StockCode,
        TRIM(Description)                        AS Product,
        SUM(Quantity)                            AS Units_Sold,
        SUM(Quantity * Price)                    AS Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0
    GROUP BY StockCode, TRIM(Description)
),

product_returned AS (
    SELECT
        StockCode,
        SUM(ABS(Quantity))                       AS Units_Returned,
        SUM(ABS(Quantity * Price))               AS Return_Value
    FROM `your_project.your_dataset.online_retail`
    WHERE STARTS_WITH(Invoice, 'C')
    GROUP BY StockCode
)

SELECT
    s.StockCode,
    s.Product,
    s.Units_Sold,
    COALESCE(r.Units_Returned, 0)               AS Units_Returned,
    ROUND(COALESCE(r.Units_Returned, 0) * 100.0
          / NULLIF(s.Units_Sold, 0), 2)         AS Return_Rate_Pct,
    ROUND(s.Revenue, 2)                          AS Gross_Revenue,
    ROUND(COALESCE(r.Return_Value, 0), 2)        AS Return_Revenue,
    ROUND(s.Revenue - COALESCE(r.Return_Value, 0), 2)
                                                 AS Net_Revenue,
    CASE
        WHEN COALESCE(r.Units_Returned, 0) * 100.0
             / NULLIF(s.Units_Sold, 0) > 20     THEN 'HIGH_RETURN — Review Quality'
        WHEN COALESCE(r.Units_Returned, 0) * 100.0
             / NULLIF(s.Units_Sold, 0) > 10     THEN 'MODERATE_RETURN — Monitor'
        ELSE                                          'ACCEPTABLE_RETURN'
    END                                          AS Action_Flag
FROM product_sold s
LEFT JOIN product_returned r USING (StockCode)
WHERE s.Units_Sold >= 50                         -- Focus on meaningful volume
ORDER BY Return_Rate_Pct DESC
LIMIT 30;


-- ─────────────────────────────────────────────────────────────
-- QUERY 16: TIME-TO-SECOND-PURCHASE BY FIRST PRODUCT CATEGORY
-- Which first purchase leads to fastest repeat buying?
-- ─────────────────────────────────────────────────────────────

WITH ordered_purchases AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        DATE(InvoiceDate)                        AS Purchase_Date,
        Invoice,
        TRIM(Description)                        AS Product,
        ROW_NUMBER() OVER (
            PARTITION BY CustomerID
            ORDER BY InvoiceDate
        )                                        AS Purchase_Rank
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY CustomerID, DATE(InvoiceDate)
        ORDER BY InvoiceDate
    ) = 1                                        -- One row per customer-day
),

first_and_second AS (
    SELECT
        f.CustomerID,
        f.Product                                AS First_Product,
        f.Purchase_Date                          AS First_Purchase_Date,
        s.Purchase_Date                          AS Second_Purchase_Date,
        DATE_DIFF(s.Purchase_Date, f.Purchase_Date, DAY)
                                                 AS Days_To_Second_Purchase
    FROM ordered_purchases f
    JOIN ordered_purchases s
        ON  f.CustomerID = s.CustomerID
        AND f.Purchase_Rank = 1
        AND s.Purchase_Rank = 2
)

SELECT
    First_Product,
    COUNT(*)                                     AS Customers,
    ROUND(AVG(Days_To_Second_Purchase), 1)       AS Avg_Days_To_Second_Purchase,
    MIN(Days_To_Second_Purchase)                 AS Min_Days,
    MAX(Days_To_Second_Purchase)                 AS Max_Days,
    ROUND(STDDEV(Days_To_Second_Purchase), 1)    AS Std_Days
FROM first_and_second
GROUP BY First_Product
HAVING COUNT(*) >= 10                            -- Meaningful sample size
ORDER BY Avg_Days_To_Second_Purchase ASC         -- Fastest repeat buyers first
LIMIT 30;


-- ─────────────────────────────────────────────────────────────
-- QUERY 17: BULK BUYER IDENTIFICATION
-- Customers with avg quantity >3 std deviations above mean
-- ─────────────────────────────────────────────────────────────

WITH global_stats AS (
    SELECT
        AVG(Quantity)                            AS Global_Avg_Qty,
        STDDEV(Quantity)                         AS Global_Std_Qty
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND CustomerID IS NOT NULL
),

customer_avg_qty AS (
    SELECT
        CAST(CustomerID AS INT64)                AS CustomerID,
        ROUND(AVG(Quantity), 2)                  AS Avg_Qty_Per_Line,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue,
        COUNT(DISTINCT Invoice)                  AS Total_Invoices,
        COUNT(*)                                 AS Total_Line_Items
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID
)

SELECT
    c.CustomerID,
    c.Avg_Qty_Per_Line,
    c.Total_Revenue,
    c.Total_Invoices,
    g.Global_Avg_Qty,
    g.Global_Std_Qty,
    ROUND((c.Avg_Qty_Per_Line - g.Global_Avg_Qty) / NULLIF(g.Global_Std_Qty, 0), 2)
                                                 AS Z_Score_Qty,
    CASE
        WHEN (c.Avg_Qty_Per_Line - g.Global_Avg_Qty)
             / NULLIF(g.Global_Std_Qty, 0) > 3  THEN 'BULK_BUYER_WHOLESALE'
        WHEN (c.Avg_Qty_Per_Line - g.Global_Avg_Qty)
             / NULLIF(g.Global_Std_Qty, 0) > 1.5 THEN 'LARGE_BUYER'
        ELSE                                          'STANDARD_RETAIL'
    END                                          AS Buyer_Type
FROM customer_avg_qty c
CROSS JOIN global_stats g
ORDER BY Z_Score_Qty DESC;

-- Revenue comparison: Bulk vs Standard buyers
WITH global_stats AS (
    SELECT AVG(Quantity) AS G_Avg, STDDEV(Quantity) AS G_Std
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice,'C') AND Quantity > 0 AND Price > 0
),
buyer_classification AS (
    SELECT
        CAST(CustomerID AS INT64) AS CustomerID,
        AVG(Quantity) AS Avg_Qty,
        SUM(Quantity*Price) AS Revenue,
        COUNT(DISTINCT Invoice) AS Frequency
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice,'C') AND Quantity>0 AND Price>0
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID
)
SELECT
    CASE WHEN (b.Avg_Qty - g.G_Avg)/NULLIF(g.G_Std,0) > 3
              THEN 'Bulk_Buyer'
         ELSE 'Standard_Buyer' END               AS Buyer_Type,
    COUNT(*)                                     AS Customer_Count,
    ROUND(AVG(Revenue), 2)                       AS Avg_Revenue,
    ROUND(AVG(Frequency), 1)                     AS Avg_Frequency,
    ROUND(AVG(Revenue/NULLIF(Frequency,0)), 2)   AS Avg_Order_Value
FROM buyer_classification b
CROSS JOIN global_stats g
GROUP BY Buyer_Type;


-- ─────────────────────────────────────────────────────────────
-- QUERY 18: MONTH-OVER-MONTH REVENUE GROWTH BY COUNTRY
-- With >20% decline flagging
-- ─────────────────────────────────────────────────────────────

WITH monthly_revenue AS (
    SELECT
        Country,
        DATE_TRUNC(DATE(InvoiceDate), MONTH)     AS Revenue_Month,
        ROUND(SUM(Quantity * Price), 2)          AS Monthly_Revenue
    FROM `your_project.your_dataset.online_retail`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY Country, DATE_TRUNC(DATE(InvoiceDate), MONTH)
),

with_mom AS (
    SELECT
        Country,
        Revenue_Month,
        Monthly_Revenue,
        LAG(Monthly_Revenue) OVER (
            PARTITION BY Country
            ORDER BY Revenue_Month
        )                                        AS Prev_Month_Revenue
    FROM monthly_revenue
)

SELECT
    Country,
    Revenue_Month,
    Monthly_Revenue,
    Prev_Month_Revenue,
    ROUND((Monthly_Revenue - Prev_Month_Revenue)
          / NULLIF(Prev_Month_Revenue, 0) * 100, 2)
                                                 AS MoM_Growth_Pct,
    CASE
        WHEN (Monthly_Revenue - Prev_Month_Revenue)
             / NULLIF(Prev_Month_Revenue, 0) < -0.20 THEN '🚨 CRITICAL_DECLINE >20%'
        WHEN (Monthly_Revenue - Prev_Month_Revenue)
             / NULLIF(Prev_Month_Revenue, 0) < 0     THEN '⚠️ DECLINING'
        WHEN (Monthly_Revenue - Prev_Month_Revenue)
             / NULLIF(Prev_Month_Revenue, 0) > 0.20  THEN '✅ STRONG_GROWTH >20%'
        ELSE                                               '➡️ STABLE'
    END                                          AS Revenue_Flag
FROM with_mom
WHERE Prev_Month_Revenue IS NOT NULL
ORDER BY Revenue_Month, MoM_Growth_Pct ASC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 19: SEGMENT REVENUE CONTRIBUTION SUMMARY
-- Business-ready stakeholder report
-- ─────────────────────────────────────────────────────────────

WITH rfm_calculated AS (
    SELECT
        CAST(CustomerID AS INT64)               AS CustomerID,
        DATE_DIFF(DATE '2011-12-10',
                  DATE(MAX(InvoiceDate)), DAY)  AS Recency,
        COUNT(DISTINCT Invoice)                 AS Frequency,
        ROUND(SUM(Quantity * Price), 2)         AS Monetary
    FROM `your_project.your_dataset.online_retail`
    WHERE CustomerID IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY CustomerID
),
scored AS (
    SELECT
        CustomerID, Recency, Frequency, Monetary,
        6 - NTILE(5) OVER (ORDER BY Recency ASC)        AS R,
        NTILE(5) OVER (ORDER BY Frequency ASC)          AS F,
        NTILE(5) OVER (ORDER BY Monetary ASC)           AS M
    FROM rfm_calculated
),
segmented AS (
    SELECT *,
        CASE
            WHEN R>=4 AND F>=4 AND M>=4          THEN 'Champions'
            WHEN R>=3 AND F>=3                   THEN 'Loyal_Customers'
            WHEN R>=4 AND F<=2                   THEN 'Recent_Customers'
            WHEN R>=3 AND M>=3                   THEN 'Potential_Loyalists'
            WHEN R=5                             THEN 'New_Customers'
            WHEN R<=2 AND F>=3                   THEN 'At_Risk'
            WHEN R<=2 AND F<=2 AND M>=3          THEN 'Cannot_Lose_Them'
            WHEN R<=2 AND F<=2                   THEN 'Lost'
            ELSE                                      'Hibernating'
        END AS Segment
    FROM scored
)
SELECT
    Segment,
    COUNT(*)                                     AS Customer_Count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1)
                                                 AS Customer_Pct,
    ROUND(SUM(Monetary), 2)                      AS Total_Revenue,
    ROUND(SUM(Monetary) * 100.0 / SUM(SUM(Monetary)) OVER(), 1)
                                                 AS Revenue_Pct,
    ROUND(AVG(Monetary), 2)                      AS Avg_Revenue_Per_Customer,
    ROUND(AVG(Frequency), 1)                     AS Avg_Orders,
    ROUND(AVG(Recency), 0)                       AS Avg_Days_Since_Last_Purchase,
    -- Business action recommendation
    CASE Segment
        WHEN 'Champions'          THEN 'VIP Rewards + Referral Program'
        WHEN 'Loyal_Customers'    THEN 'Upsell Premium Products + Loyalty Points'
        WHEN 'At_Risk'            THEN 'Win-Back Email at 45-Day Inactivity'
        WHEN 'Lost'               THEN 'Final Re-engagement Offer or Deprioritize'
        WHEN 'New_Customers'      THEN 'Onboarding Series + 2nd Purchase Incentive'
        WHEN 'Potential_Loyalists' THEN 'Personalized Recommendations'
        WHEN 'Recent_Customers'   THEN 'Engagement Campaign to Build Frequency'
        WHEN 'Cannot_Lose_Them'   THEN 'Personal Outreach + Special Offer'
        ELSE                           'Standard Campaign'
    END                                          AS Recommended_Action
FROM segmented
GROUP BY Segment
ORDER BY Total_Revenue DESC;

-- ============================================================
-- END OF SQL ANALYSIS
-- Total queries: 19 analytical queries
-- Covers: Data Quality, RFM, Cohort, Product, Basket,
--         Behavior, Anomaly, Returns, Time-Series
-- ============================================================
