# Customer Segmentation — Complete Analysis Guide
## UCI Online Retail II Dataset | 1M+ Transactions

---

## HOW TO RUN

### Step 1: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
pip install lifelines lifetimes mlxtend umap-learn plotly xgboost
```

### Step 2: Folder Structure
```
project/
├── online_retail_II.xlsx          ← Dataset
├── customer_segmentation_python.py ← All Python code
├── customer_segmentation_sql.sql   ← All SQL queries
├── models/                         ← Saved models
└── output/                         ← All charts & CSVs (auto-created)
```

### Step 3: Run Python Analysis
```bash
python customer_segmentation_python.py
```

### Step 4: Run SQL Queries
- Upload dataset to BigQuery as `online_retail`
- Run queries from `customer_segmentation_sql.sql`
- Replace `your_project.your_dataset.online_retail` with your table path

---

## COMPLETE ANALYSIS COVERAGE

### PYTHON SECTIONS (20 Sections)

| Section | Analysis | Key Output |
|---------|----------|------------|
| 2 | Data Quality Audit (5 row types) | data_quality_audit.png |
| 3 | Missing Customer ID Pattern | Console output |
| 4 | Partial Month Bias Correction | Reference date fix |
| 5 | Return Behavior Analysis | top_returned_products.png |
| 6 | Advanced RFM (10 features) | feature_enrichment_silhouette.png |
| 7 | Rolling Quarterly RFM | quarterly_rfm_trend.png |
| 8 | Weighted RFM Scoring | rfm_segments.png |
| 9 | 4-Algorithm Clustering Comparison | clustering_comparison.png |
| 10 | Cluster Stability (50 seeds, ARI) | cluster_stability.png |
| 11 | PCA vs UMAP Visualization | pca_umap_visualization.png |
| 12 | Pareto Revenue Analysis | pareto_analysis.png |
| 13 | Segment Migration Matrix (Q1→Q4) | segment_migration_matrix.png |
| 14 | Kaplan-Meier Survival Curve | survival_curve.png |
| 15 | One-Time → Repeat Buyer Prediction | one_time_buyer_prediction.png |
| 16 | BG/NBD + Gamma-Gamma CLV | clv_distribution.png |
| 17 | Apriori Market Basket Analysis | association_rules.png |
| 18 | Premium Signal Products | premium_signal_products.png |
| 19 | Hypothesis Testing (3 tests) | Console output |
| 20 | Business Action Summary | final_rfm_segmentation.csv |

### SQL QUERIES (19 Queries)

| Query | Analysis | Key Technique |
|-------|----------|---------------|
| 1 | Data Quality Audit | CASE classification |
| 2 | Missing ID Pattern | GROUP BY comparison |
| 3 | Full RFM in SQL | NTILE window function |
| 4 | Pareto Analysis | Cumulative SUM OVER |
| 5 | Cohort Retention Table | DATE_DIFF + pivot |
| 6 | Churned Customer ID | Half-period comparison |
| 7 | Inter-Purchase Intervals | LAG + STDDEV |
| 8 | SQL Market Basket (Lift) | Self-JOIN + lift formula |
| 9 | Revenue Per Customer/Product | Ratio calculation |
| 10 | Declining Sales Velocity | Quarterly CASE pivot |
| 11 | Country Percentile Analysis | PERCENT_RANK PARTITION |
| 12 | Anomalous Order Detection | Z-Score with STDDEV |
| 13 | Day-Hour Revenue Heatmap | EXTRACT + FORMAT_DATE |
| 14 | Return Matching Net Revenue | LEFT JOIN cancellations |
| 15 | Product Return Rate | Units ratio |
| 16 | Time-to-Second-Purchase | ROW_NUMBER + DATE_DIFF |
| 17 | Bulk Buyer Identification | Global Z-Score |
| 18 | Month-over-Month Growth | LAG window function |
| 19 | Segment Summary Report | Full stakeholder view |

---

## KEY METRICS THIS ANALYSIS PRODUCES

- Pareto threshold: X% of customers = 80% of revenue
- Median customer survival: N days before dormancy
- Win-back trigger: Day N of inactivity (70% of median)
- Segment migration rates: % of Champions retained Q1→Q4
- One-time buyer conversion accuracy: X% (Random Forest)
- 12-month predicted CLV per customer (BG/NBD model)
- Most stable K for clustering (Adjusted Rand Index)
- Premium signal products: bought by Champions, never by Lost

---

## DIFFERENTIATORS VS STANDARD GITHUB REPOS

| Standard Notebook | This Analysis |
|------------------|---------------|
| K-Means only | 4 algorithms compared |
| 3 RFM features | 10 engineered features |
| Static RFM | Quarterly rolling RFM |
| No evaluation | ARI stability + 3 metrics |
| Cluster 0,1,2 labels | Named business segments |
| No business action | Full action plan per segment |
| No hypothesis testing | 3 statistical tests with effect sizes |
| No CLV | BG/NBD + Gamma-Gamma CLV |
| No survival analysis | Kaplan-Meier churn timing |
| No ML layer | One-time buyer classifier |
| No SQL | 19 production-quality SQL queries |

---

## INTERVIEW TALKING POINTS

1. "I compared 4 clustering algorithms using silhouette, 
   Davies-Bouldin, and Calinski-Harabasz — not just elbow curve"

2. "I ran 50-seed stability analysis to find the most 
   reproducible cluster count using Adjusted Rand Index"

3. "I built a rolling quarterly RFM to track segment migration, 
   which is what retention teams at Amazon and Flipkart actually monitor"

4. "The survival analysis gives us a precise trigger point — 
   50% of customers go dormant by day N, so we launch win-back 
   campaigns at 70% of that threshold"

5. "I quantified the Pareto principle: X% of customers drive 80% 
   of revenue, and Champions represent Y% of that group"

6. "The SQL cohort retention matrix is built entirely in BigQuery 
   using CTEs and DATE_DIFF — no Python needed"
