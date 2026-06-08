# Model Card: Customer Segmentation with RFM Analysis

---

## 1. Model Details

| Field | Details |
|---|---|
| **Framework Name** | Customer Segmentation at Scale — Multi-Algorithm RFM Framework |
| **Python Version** | 3.10 |
| **Analysis Date** | March 2026 |
| **Primary Method** | Weighted RFM Scoring (R=0.3, F=0.3, M=0.4) |
| **Supporting Methods** | K-Means, Agglomerative Clustering, GMM, DBSCAN |
| **Primary Metric** | Silhouette Score (cluster quality) |
| **Secondary Metrics** | Davies-Bouldin Score, Calinski-Harabasz Score, Adjusted Rand Index |
| **Live App** | Streamlit Interactive Dashboard |

---

## 2. Intended Use

- **Primary Use Case:** Segment 5,878 unique customers from a UK-based online retailer into actionable groups (Champions, Loyal, At Risk, Lost, etc.) using RFM analysis, advanced feature engineering, and statistical hypothesis testing.
- **Target Users:** Marketing Analysts, CRM Teams, Retention Strategists, E-commerce Data Scientists.
- **Out of Scope:** Real-time transaction scoring, product-level SKU recommendations, or customers outside the 2009–2011 dataset window.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Source File** | `online_retail_II.xlsx` |
| **Source** | [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **Sheets** | Year 2009-2010 + Year 2010-2011 (concatenated) |
| **Total Unique Customers** | 5,878 |
| **Timeframe** | 2009–2011 (weekly transactions) |
| **Reference Date** | 2011-12-10 (day after last transaction) |
| **Max Customer Monetary Value** | £608,821.60 |
| **Median Customer Monetary Value** | £898.90 |

**Data Quality Audit — Row Classification:**

| Row Type | Description |
|---|---|
| `Valid_Purchase` | Positive quantity, positive price, no cancellation flag |
| `Cancellation` | Invoice starts with 'C' |
| `Negative_Price_Adjustment` | Price < 0 (non-transactional adjustments) |
| `Free_Sample_Promotional` | Price = 0, Quantity > 0 |
| `Return_Without_C_Flag` | Negative quantity without 'C' prefix |

**Preprocessing Rules Applied:**
```
DROP rows where Customer ID is null
DROP rows where Quantity <= 0 or Price <= 0
DROP rows where Invoice starts with 'C' (cancellations)
FEATURE: TotalSum = Quantity × Price
REFERENCE DATE: max(InvoiceDate) + 1 day
```

**Dataset Schema:**

| Feature | Description | Data Type |
|---|---|---|
| `Invoice` | 6-digit transaction ID ('C' prefix = cancellation) | object |
| `StockCode` | 5-digit product code | object |
| `Description` | Product name | object |
| `Quantity` | Units per transaction | int64 |
| `InvoiceDate` | Date and time of transaction | datetime64[ns] |
| `Price` | Unit price in GBP | float64 |
| `Customer ID` | 5-digit customer identifier | float64 |
| `Country` | Customer's country of residence | object |

---

## 4. Feature Engineering

**Base RFM Features → 10 Engineered Features**

### Core RFM Metrics
| Feature | Description |
|---|---|
| `Recency` | Days since last purchase (from reference date 2011-12-10) |
| `Frequency` | Count of unique invoices |
| `MonetaryValue` | Sum of all valid purchase totals |

### Advanced Engineered Features
| Feature | Description |
|---|---|
| `Avg_Order_Value` | Mean revenue per invoice |
| `Purchase_Regularity_Std` | Std deviation of days between purchases (lower = more regular) |
| `Return_Rate` | Cancelled invoices / total invoices |
| `Product_Diversity` | Count of unique StockCodes purchased |
| `Best_Quarter` | Quarter with highest total spend |
| `Weekend_Purchase_Ratio` | Proportion of purchases on Saturday/Sunday |
| `Peak_Hour_Segment` | Modal purchase hour bucket (Morning / Afternoon / Evening / Night) |

**Feature Enrichment Validation:**
Extended RFM (8 features) achieved higher Silhouette scores than Base RFM (3 features) across all K values (K=3 to K=8), confirming that additional feature engineering improves cluster quality.

---

## 5. Methodology & Pipeline Architecture




## 6. RFM Scoring & Segmentation

### Quintile Scoring
| Dimension | Scoring Logic |
|---|---|
| `R_Score` | qcut into 5 bins; lower recency (more recent) = score 5 |
| `F_Score` | qcut on rank; higher frequency = score 5 |
| `M_Score` | qcut on rank; higher monetary = score 5 |

### Weighted RFM Score
```
Weighted_RFM = (R_Score × 0.3) + (F_Score × 0.3) + (M_Score × 0.4)
```
**Weight justification:** Correlation analysis showed Frequency and Monetary have a log-correlation of 0.85, making them the strongest predictors of customer value. The 40% weight on Monetary reflects its higher predictive power for revenue impact.

### Segment Assignment Rules
| Segment | Condition |
|---|---|
| **Champions** | R ≥ 4 AND F ≥ 4 AND M ≥ 4 |
| **Loyal_Customers** | R ≥ 3 AND F ≥ 3 |
| **Recent_Customers** | R ≥ 4 AND F ≤ 2 |
| **Potential_Loyalists** | R ≥ 3 AND M ≥ 3 |
| **New_Customers** | R = 5 |
| **At_Risk** | R ≤ 2 AND F ≥ 3 |
| **Cannot_Lose_Them** | R ≤ 2 AND F ≤ 2 AND M ≥ 3 |
| **Lost** | R ≤ 2 AND F ≤ 2 |
| **Hibernating** | All other |

---

## 7. Segment Results

**5,878 total customers segmented across 9 tiers:**

| Segment | Count | Avg Recency (days) | Avg Frequency | Avg Monetary (£) |
|---|---|---|---|---|
| Loyal_Customers | 1,412 | 71.1 | 5.4 | 1,940.90 |
| Champions | 1,294 | 19.3 | 17.1 | **9,354.30** |
| Lost | 1,274 | 467.6 | 1.2 | 256.80 |
| At_Risk | 821 | 369.3 | 4.9 | 1,986.50 |
| Recent_Customers | 441 | 27.5 | 1.5 | 894.30 |
| Hibernating | 283 | 107.9 | 1.2 | 289.60 |
| Cannot_Lose_Them | 248 | 415.0 | 1.6 | 1,368.90 |
| Potential_Loyalists | 105 | 103.1 | 1.7 | 1,185.10 |

**Overall RFM Distribution:**

| Metric | Mean | Median | Max |
|---|---|---|---|
| Recency (days) | 200.9 | 95.0 | 738 |
| Frequency (orders) | 6.3 | 3.0 | 398 |
| Monetary (£) | 3,018.60 | 898.90 | 608,821.60 |

---

## 8. Pareto Analysis

**Key finding:** The top 20% of customers generate **77.26% of total revenue**, closely following the Pareto Principle.

- Losing the Champions segment (top ~5–10%) would eliminate over 50% of stable income.
- Champions segment composition dominates the top 20% revenue tier.

---

## 9. Clustering Algorithm Comparison

All algorithms evaluated on standard-scaled base RFM (Recency, Frequency, Monetary) with K=4:

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Time (s) |
|---|---|---|---|
| **K-Means** | 0.589 | 0.586 | 0.81 |
| **Agglomerative** | **0.590** | 0.628 | 1.71 |
| GMM | 0.183 | 1.189 | 0.95 |

**Selected Model: K-Means** — chosen for computational efficiency and business interpretability despite Agglomerative's marginally higher silhouette score.

### Cluster Stability Analysis (50 seeds, K=3 to K=8)
K-Means was run 50 times with different random seeds per K value. Adjusted Rand Index (ARI) was computed against the first-seed run to measure consistency. The most stable K was selected for final PCA/UMAP visualisation.

---

## 10. Hypothesis Testing Results

### Test 1 — Mann-Whitney U: Champions vs At-Risk Monetary Value
- **Result:** p < 0.001, Cohen's d = 0.563
- **Conclusion:** ✅ Champions have statistically significantly higher monetary value than At-Risk customers. Segmentation logic effectively separates high-value and low-value groups.

### Test 2 — Kruskal-Wallis: Country RFM Profile Differences
- **Result:** p < 0.05
- **Conclusion:** ✅ Country significantly affects RFM profile. UK dominates volume; EIRE and Netherlands show higher average spend per customer.

### Test 3 — Chi-Square: Season of First Purchase → Long-Term Segment
- **Result:** p < 0.05
- **Conclusion:** ✅ The quarter of a customer's first purchase significantly predicts their long-term segment membership.

---

## 11. Customer Lifetime Value (BG/NBD + Gamma-Gamma)

The framework supports CLV prediction using probabilistic models (requires `lifetimes` library):

- **BG/NBD Model:** Predicts expected purchases in next 90 days per customer
- **Gamma-Gamma Model:** Estimates expected revenue per future purchase
- **Output:** 12-month predicted CLV per customer

> Prerequisite: `pip install lifetimes`

---

## 12. Survival Analysis (Kaplan-Meier)

- **Churn definition:** Last purchase > 90 days before reference date
- **Output:** Median customer survival time in days
- **Business action:** Trigger win-back campaign at 70% of median survival time

> Prerequisite: `pip install lifelines`

---

## 13. Quarterly Segment Migration (Q1 → Q4 2010)

Tracks customer movement between segments across quarters using a migration matrix. Key outputs:
- Retention rate per segment (% staying in same segment Q1 → Q4)
- Improving vs. declining customer counts by score delta

---

## 14. Business Action Plan by Segment

| Segment | Strategy | Action | Target Metric |
|---|---|---|---|
| **Champions** | Reward & Retain | VIP loyalty program, early product access, referral incentives | Maintain >85% retention rate |
| **Loyal_Customers** | Upsell & Cross-sell | Premium recommendations, bundle offers | Increase AOV by 15% |
| **At_Risk** | Win-Back Campaign | Trigger email at 45-day inactivity with 10% discount | Recover 25% of At-Risk customers |
| **Lost** | Re-engage or Deprioritize | Final win-back offer; remove from active campaigns if no response | Cost savings from reduced marketing spend |
| **New_Customers** | Onboarding & Education | Welcome series, 2nd purchase incentive within 30 days | Convert 40% to Loyal within 90 days |
| **Potential_Loyalists** | Nurture to Loyalty | Personalised recommendations, loyalty points | Move 30% to Loyal within 6 months |
| **Cannot_Lose_Them** | Personal Outreach | Direct contact + special offer for high-value dormant customers | Reactivate 20% within 60 days |
| **Hibernating** | Low-cost Re-engagement | Seasonal email, new product alerts | Reactivate at minimal campaign cost |

---

## 15. Ethical Considerations & Limitations

- **Temporal Drift:** The dataset covers 2009–2011. UK retail patterns have changed significantly — retraining on recent data is essential before production use.
- **Guest Transactions:** Transactions with missing Customer IDs cannot be assigned to segments, creating an invisible customer pool. These guests show lower AOV and may represent an untapped conversion opportunity.
- **Negative Sales / Returns:** Return rates are used as a feature but high-frequency buyers also tend to have higher return rates. Net monetary value for these customers remains positive.
- **Negative Price Entries:** Values as low as -£53,594 were identified as non-transactional adjustments. These must be excluded from CLV models to avoid underestimation.
- **Country Bias:** UK dominates transaction volume (~90%+). Segment thresholds may not generalise well to other countries with smaller sample sizes.
- **Partial Months:** First and last months of the dataset may be partial — reference date selection (2011-12-10) mitigates this for recency calculations.
- **Single-Channel:** Data covers only online retail transactions; in-store or phone purchases are not represented, potentially misclassifying omnichannel customers as lower-value.

---

## 16. Infrastructure & Tools

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Clustering | Scikit-learn (K-Means, DBSCAN, AgglomerativeClustering, GMM) |
| Dimensionality Reduction | PCA (scikit-learn), UMAP (umap-learn, optional) |
| Statistical Tests | SciPy (Mann-Whitney U, Kruskal-Wallis, Chi-Square), Statsmodels (Tukey HSD) |
| Survival Analysis | lifelines (KaplanMeierFitter) — optional |
| CLV Prediction | lifetimes (BG/NBD, Gamma-Gamma) — optional |
| Market Basket | mlxtend (Apriori, Association Rules) — optional |
| Visualisation | Matplotlib, Seaborn |
| SQL Engine | Google BigQuery (BigQuery Studio) |
| Frontend | Streamlit |
| Experiment Tracking | MLflow (SQLite backend) |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Containerisation | Docker |
| Cloud Infrastructure | AWS EC2 |
| Version Control | Git |
---

## 17. Final Decision Summary

```
══════════════════════════════════════════════════════════════
        CUSTOMER SEGMENTATION — EXECUTIVE SUMMARY REPORT
══════════════════════════════════════════════════════════════
Dataset:         Online Retail II | UK-based, 2009–2011
Unique Customers: 5,878 | Reference Date: 2011-12-10
Primary Method:  Weighted RFM (R=0.3, F=0.3, M=0.4)
══════════════════════════════════════════════════════════════
SEGMENTATION OUTPUT:
Champions:          1,294  | Avg Monetary: £9,354
Loyal_Customers:    1,412  | Avg Monetary: £1,941
At_Risk:              821  | Avg Monetary: £1,987
Lost:               1,274  | Avg Monetary: £257
══════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS:
1. Weighted RFM (not equal weights) — M weighted at 40%
2. 10 engineered features — validated via silhouette improvement
3. 4-algorithm comparison — K-Means selected for efficiency
4. 50-seed stability test — confirms cluster reproducibility
5. Hypothesis testing — statistically validates segment differences
══════════════════════════════════════════════════════════════
PRODUCTION RECOMMENDATIONS:
• Retrain quarterly as new transaction data arrives
• Trigger win-back campaign at 45-day customer inactivity
• Prioritise Champions segment for referral programmes
• Convert Guest (missing ID) transactions via account incentives
• Log all segmentation runs to MLflow for audit trail
• Monitor Champions → At_Risk migration rate monthly
══════════════════════════════════════════════════════════════
```
