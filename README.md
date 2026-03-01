# Customer Segmentation with RFM Analysis

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project involves a comprehensive analysis of the "Online Retail II" dataset, which contains transactions for a UK-based non-store online retail from 2009 to 2011. The goal is to perform advanced customer segmentation using RFM (Recency, Frequency, Monetary) analysis, advanced feature engineering, and multiple clustering algorithms to identify high-value customers and at-risk segments.

### Executive Summary

The analysis identified that the top 20% of customers generate 77.26% of total revenue, closely following the Pareto Principle. We uncovered significant data quality issues, including anomalous negative prices and partial month biases. By engineering advanced features like Purchase Regularity and Return Rate, we differentiated standard buyers from wholesale and erratic shoppers. A Weighted RFM Model (30% R, 30% F, 40% M) proved most effective for segmenting the 5,942 unique customers into actionable groups such as 'Champions', 'At Risk', and 'Loyal'.

### Goal

To build a production-grade customer segmentation pipeline that goes beyond basic K-Means by incorporating return behavior analysis, seasonality, and statistical hypothesis testing to drive targeted marketing strategies.

### Data structure and initial checks
[Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| Invoice  | Unique 6-digit identifier for each transaction (Starts with 'C' for cancellations) | object | 
| StockCode | Unique 5-digit product code | object | 
| Description | Product name | object | 
| Quantity   | Quantities of each product per transaction | int64 | 
| InvoiceDate | The day and time when each transaction was generated | datetime64[ns] | 
| Price | Unit price of the product | float64 | 
| Customer ID | Unique 5-digit identifier for each customer | float64 | 
| Country | Name of the country where each customer resides | object | 

### Tools

1). Excel/CSV: Initial data inspection and output storage.

2). SQL: Used for production-ready queries including Cohort Analysis, Window Functions for Pareto thresholds, and Rolling Retention.

3). Python: Used for data cleaning, advanced feature engineering, and machine learning. Libraries: Pandas, Numpy, Scikit-learn (K-Means, GMM, Agglomerative), Scipy (Stats), Matplotlib, Seaborn.

### Analysis
**Python**

Laoding all the necessay libraries
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, adjusted_rand_score

# Stats
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
```
``` python
# Survival Analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines import BetaGeoFitter, GammaGammaFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Install lifelines: pip install lifelines")
```
``` python
# Market Basket
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("Install mlxtend: pip install mlxtend")
```
``` python
# UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Install umap: pip install umap-learn")

import time
```

Laoding the complete dataset

``` python
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Loading both the sheets of the UCI Online Retail II dataset, Sheet 1 = Year 2009-2010, Sheet 2 = Year 2010-2011
try:
    df_1 = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2009-2010')
    df_2 = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
    df = pd.concat([df_1, df_2], ignore_index=True)
    print(f"Combined dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
except:
    # If only one sheet available
    df = pd.read_excel('online_retail_II.xlsx')
    print(f"Single sheet loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
```

<img width="427" height="80" alt="image" src="https://github.com/user-attachments/assets/2db2a029-ebd9-418e-be6f-34c26cbe2e6b" />

``` python
# Standardize column names
df.columns = df.columns.str.strip()
print(df.head())
print(df.dtypes)
```

<img width="470" height="363" alt="image" src="https://github.com/user-attachments/assets/65465317-3c28-4d58-80e7-263d0b90c078" />

A. Data Quality & Advanced Preprocessing

Comphrensive Data Quality Audit

``` python
print("\n" + "=" * 60)
print("SECTION 2: COMPREHENSIVE DATA QUALITY AUDIT")
print("=" * 60)

def classify_row(row):
    """Classify each transaction into one of 5 states."""
    invoice = str(row.get('Invoice', ''))
    qty     = row.get('Quantity', 0)
    price   = row.get('Price', 0)

    if invoice.startswith('C'):
        return 'Cancellation'
    elif price < 0:
        return 'Negative_Price_Adjustment'
    elif price == 0 and qty > 0:
        return 'Free_Sample_Promotional'
    elif qty < 0 and not invoice.startswith('C'):
        return 'Return_Without_C_Flag'
    else:
        return 'Valid_Purchase'

df['Row_Type'] = df.apply(classify_row, axis=1)

# Revenue impact per category
df['Revenue'] = df['Quantity'] * df['Price']
audit = df.groupby('Row_Type').agg(
    Row_Count=('Invoice', 'count'),
    Total_Revenue=('Revenue', 'sum'),
    Avg_Price=('Price', 'mean'),
    Avg_Qty=('Quantity', 'mean')
).reset_index()
audit['Row_Pct']     = (audit['Row_Count']   / len(df) * 100).round(2)
audit['Revenue_Pct'] = (audit['Total_Revenue'] / df['Revenue'].sum() * 100).round(2)

print("\nData Quality Audit Report:")
print(audit.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].pie(audit['Row_Count'], labels=audit['Row_Type'],
            autopct='%1.1f%%', startangle=140)
axes[0].set_title('Row Distribution by Type')
audit_pos = audit[audit['Total_Revenue'] > 0]
axes[1].bar(audit_pos['Row_Type'], audit_pos['Revenue_Pct'], color='steelblue')
axes[1].set_title('Revenue Contribution by Row Type (%)')
axes[1].set_ylabel('Revenue %')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
```

<img width="653" height="197" alt="image" src="https://github.com/user-attachments/assets/9d17a688-1be9-4b0e-9e63-b0950e082bbf" />

<img width="1328" height="490" alt="image" src="https://github.com/user-attachments/assets/5f114908-baa8-48a3-a28e-138edfa80528" />

Missing Custommer ID's Pattern Analysis

``` python
print("\n" + "=" * 60)
print("SECTION 3: MISSING CUSTOMER ID PATTERN ANALYSIS")
print("=" * 60)

df['Has_CustomerID'] = df['Customer ID'].notna()

# Compare purchasing patterns
missing_analysis = df.groupby('Has_CustomerID').agg(
    Count=('Invoice', 'count'),
    Avg_Order_Value=('Revenue', 'mean'),
    Avg_Qty=('Quantity', 'mean'),
    Unique_Countries=('Country', 'nunique'),
    Avg_Hour=('InvoiceDate', lambda x: pd.to_datetime(x).dt.hour.mean())
).reset_index()
missing_analysis['Has_CustomerID'] = missing_analysis['Has_CustomerID'].map(
    {True: 'Has_ID', False: 'Missing_ID'}
)
print("\nPurchase Pattern Comparison (Missing vs Present Customer ID):")
print(missing_analysis.to_string(index=False))

# Country distribution for missing IDs
missing_country = df[~df['Has_CustomerID']]['Country'].value_counts().head(10)
print("\nTop 10 Countries with Missing Customer IDs:")
print(missing_country)

# Decision: Drop missing Customer IDs (they cannot be used for RFM)
df_clean = df[
    (df['Has_CustomerID']) &
    (df['Row_Type'] == 'Valid_Purchase') &
    (df['Quantity'] > 0) &
    (df['Price'] > 0)
].copy()
print(f"\nClean dataset after filtering: {df_clean.shape[0]:,} rows")

```
<img width="523" height="424" alt="image" src="https://github.com/user-attachments/assets/c4ec2ba2-e84d-418f-819a-ebd8c5856f83" />

Partial Month Bias Correction

``` python
print("\n" + "=" * 60)
print("SECTION 4: PARTIAL MONTH BIAS CORRECTION")
print("=" * 60)

df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['YearMonth']   = df_clean['InvoiceDate'].dt.to_period('M')

# Monthly transaction counts
monthly_counts = df_clean.groupby('YearMonth')['Invoice'].count()
print("\nMonthly Transaction Counts:")
print(monthly_counts)

# Flag partial months
first_month = monthly_counts.index.min()
last_month  = monthly_counts.index.max()
print(f"\nFirst month: {first_month} | Last month: {last_month}")
print("NOTE: First and last months may be partial — verify counts vs typical months")

# Use a reference date that avoids partial month bias
# Use the last day of the second-to-last complete month
REFERENCE_DATE = pd.Timestamp('2011-12-09') + pd.Timedelta(days=1)
print(f"\nReference Date for Recency calculation: {REFERENCE_DATE.date()}")

# Filter to exclude partial first month if needed
# df_clean = df_clean[df_clean['YearMonth'] > first_month]
```

<img width="411" height="487" alt="image" src="https://github.com/user-attachments/assets/adad7d2a-17e0-4d90-970c-c7a84cad8e7a" />
<img width="515" height="131" alt="image" src="https://github.com/user-attachments/assets/de0c6afc-c723-48d3-8d72-dd2e5a84a654" />

Return Behaviour Analysis

``` python
print("\n" + "=" * 60)
print("SECTION 5: RETURN BEHAVIOR ANALYSIS")
print("=" * 60)

returns = df[df['Row_Type'] == 'Cancellation'].copy()
returns['Customer ID'] = returns['Customer ID'].astype(str)

# Top returning customers
top_returners = returns.groupby('Customer ID').agg(
    Return_Count=('Invoice', 'count'),
    Total_Return_Value=('Revenue', 'sum'),
    Unique_Products_Returned=('StockCode', 'nunique')
).sort_values('Return_Count', ascending=False).head(20)
print("\nTop 20 Customers by Return Count:")
print(top_returners)

# Top returned products
top_returned_products = returns.groupby('Description').agg(
    Return_Count=('Quantity', lambda x: abs(x).sum()),
    Return_Revenue=('Revenue', 'sum')
).sort_values('Return_Count', ascending=False).head(15)
print("\nTop 15 Most Returned Products:")
print(top_returned_products)

# Visualize
fig, ax = plt.subplots(figsize=(12, 5))
top_returned_products['Return_Count'].plot(kind='barh', ax=ax, color='salmon')
ax.set_title('Top 15 Most Returned Products')
ax.set_xlabel('Units Returned')
plt.tight_layout()
plt.show()
```

<img width="520" height="460" alt="image" src="https://github.com/user-attachments/assets/8257b30c-d7e7-4c7e-900c-c99b0b237a36" />

<img width="452" height="303" alt="image" src="https://github.com/user-attachments/assets/780abfe8-c04f-4cb8-8126-a84dcbba8197" />

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/2b7c5132-55ec-4a8b-a462-a27158f8deaa" />


B. Advanced RFM Feature Engineering

Advanced RFM feature Engineering

``` python

print("\n" + "=" * 60)
print("SECTION 6: ADVANCED RFM FEATURE ENGINEERING (10 Features)")
print("=" * 60)

df_rfm = df_clean.copy()
df_rfm['Customer ID'] = df_rfm['Customer ID'].astype(int)
df_rfm['TotalPrice']  = df_rfm['Quantity'] * df_rfm['Price']
df_rfm['InvoiceDate'] = pd.to_datetime(df_rfm['InvoiceDate'])
df_rfm['DayOfWeek']   = df_rfm['InvoiceDate'].dt.dayofweek   # 0=Mon, 6=Sun
df_rfm['Hour']        = df_rfm['InvoiceDate'].dt.hour
df_rfm['Quarter']     = df_rfm['InvoiceDate'].dt.quarter

# Base RFM 
snapshot_date = REFERENCE_DATE

base_rfm = df_rfm.groupby('Customer ID').agg(
    Last_Purchase=('InvoiceDate', 'max'),
    Frequency=('Invoice', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()
base_rfm['Recency'] = (snapshot_date - base_rfm['Last_Purchase']).dt.days

# Feature 1: Average Order Value 
aov = df_rfm.groupby('Customer ID').apply(
    lambda x: x.groupby('Invoice')['TotalPrice'].sum().mean()
).reset_index()
aov.columns = ['Customer ID', 'Avg_Order_Value']

# Feature 2: Purchase Regularity (std of days between purchases) 
def purchase_regularity(group):
    dates = group['InvoiceDate'].dt.normalize().drop_duplicates().sort_values()
    if len(dates) < 2:
        return 0
    diffs = dates.diff().dropna().dt.days
    return diffs.std() if len(diffs) > 1 else 0

regularity = df_rfm.groupby('Customer ID').apply(purchase_regularity).reset_index()
regularity.columns = ['Customer ID', 'Purchase_Regularity_Std']

# Feature 3: Return Rate 
all_invoices = df[df['Has_CustomerID']].copy()
all_invoices['Customer ID'] = all_invoices['Customer ID'].astype(int)
cancelled   = all_invoices[all_invoices['Row_Type'] == 'Cancellation']
total_inv   = all_invoices.groupby('Customer ID')['Invoice'].nunique().reset_index()
total_inv.columns = ['Customer ID', 'Total_Invoices']
cancel_inv  = cancelled.groupby('Customer ID')['Invoice'].nunique().reset_index()
cancel_inv.columns = ['Customer ID', 'Cancel_Invoices']
return_rate = total_inv.merge(cancel_inv, on='Customer ID', how='left').fillna(0)
return_rate['Return_Rate'] = (
    return_rate['Cancel_Invoices'] / return_rate['Total_Invoices']
).round(4)

# Feature 4: Product Diversity 
diversity = df_rfm.groupby('Customer ID')['StockCode'].nunique().reset_index()
diversity.columns = ['Customer ID', 'Product_Diversity']

# Feature 5: Seasonal Preference (best quarter) 
seasonal = df_rfm.groupby(['Customer ID', 'Quarter'])['TotalPrice'].sum().reset_index()
best_quarter = seasonal.loc[seasonal.groupby('Customer ID')['TotalPrice'].idxmax()]
best_quarter = best_quarter[['Customer ID', 'Quarter']].rename(
    columns={'Quarter': 'Best_Quarter'}
)

# Feature 6: Weekend vs Weekday Purchase Ratio 
df_rfm['Is_Weekend'] = df_rfm['DayOfWeek'].isin([5, 6]).astype(int)
weekend_ratio = df_rfm.groupby('Customer ID').apply(
    lambda x: x['Is_Weekend'].mean()
).reset_index()
weekend_ratio.columns = ['Customer ID', 'Weekend_Purchase_Ratio']

# Feature 7: Peak Hour Preference (Morning/Afternoon/Evening/Night) ──
def hour_segment(h):
    if   5  <= h < 12: return 'Morning'
    elif 12 <= h < 17: return 'Afternoon'
    elif 17 <= h < 21: return 'Evening'
    else:              return 'Night'

df_rfm['Hour_Segment'] = df_rfm['Hour'].apply(hour_segment)
peak_hour = df_rfm.groupby('Customer ID')['Hour_Segment'].agg(
    lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
).reset_index()
peak_hour.columns = ['Customer ID', 'Peak_Hour_Segment']

rfm_full = (base_rfm
    .merge(aov,          on='Customer ID', how='left')
    .merge(regularity,   on='Customer ID', how='left')
    .merge(return_rate[['Customer ID', 'Return_Rate']], on='Customer ID', how='left')
    .merge(diversity,    on='Customer ID', how='left')
    .merge(best_quarter, on='Customer ID', how='left')
    .merge(weekend_ratio,on='Customer ID', how='left')
    .merge(peak_hour,    on='Customer ID', how='left')
)
rfm_full = rfm_full.fillna(0)

print(f"\nFull RFM Feature Matrix: {rfm_full.shape}")
print(rfm_full.describe())
print(rfm_full.head())
```

<img width="510" height="408" alt="image" src="https://github.com/user-attachments/assets/a1dd2a90-4cec-4063-8005-9377da0f6854" />

<img width="530" height="486" alt="image" src="https://github.com/user-attachments/assets/595ebd9e-9bca-4df4-aac1-e889c9a813cb" />

Comparisons of silhouette scores of base RFM vs extended RFM 

``` python
# Compare silhouette scores: base RFM vs extended RFM 
scaler = StandardScaler()
base_features   = ['Recency', 'Frequency', 'Monetary']
extended_features = base_features + [
    'Avg_Order_Value', 'Purchase_Regularity_Std',
    'Return_Rate', 'Product_Diversity', 'Weekend_Purchase_Ratio'
]

X_base     = scaler.fit_transform(rfm_full[base_features])
X_extended = scaler.fit_transform(rfm_full[extended_features])

sil_base, sil_ext = [], []
K_range = range(3, 9)
for k in K_range:
    km_b = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_base)
    km_e = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_extended)
    sil_base.append(silhouette_score(X_base,     km_b.labels_))
    sil_ext.append( silhouette_score(X_extended, km_e.labels_))

plt.figure(figsize=(9, 4))
plt.plot(K_range, sil_base, 'o--', label='Base RFM (3 features)')
plt.plot(K_range, sil_ext,  's-',  label='Extended RFM (8 features)')
plt.xlabel('Number of Clusters K')
plt.ylabel('Silhouette Score')
plt.title('Feature Enrichment Impact on Clustering Quality')
plt.legend()
plt.tight_layout()
plt.show()
print("\nConclusion: Higher silhouette on extended features validates the additional engineering.")
```

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/5236a718-a161-455e-84af-73ff69dfaf57" />

<img width="608" height="34" alt="image" src="https://github.com/user-attachments/assets/faa176b9-9fce-4aee-8fab-a55ae31e7203" />

Rolling RFM of Quaterly tracking

``` python

print("\n" + "=" * 60)
print("SECTION 7: ROLLING RFM — QUARTERLY CUSTOMER TRACKING")
print("=" * 60)

quarters = {
    'Q1_2010': ('2010-01-01', '2010-03-31'),
    'Q2_2010': ('2010-04-01', '2010-06-30'),
    'Q3_2010': ('2010-07-01', '2010-09-30'),
    'Q4_2010': ('2010-10-01', '2010-12-31'),
}

quarterly_rfm = {}
for qname, (start, end) in quarters.items():
    ref = pd.Timestamp(end) + pd.Timedelta(days=1)
    q_df = df_rfm[
        (df_rfm['InvoiceDate'] >= pd.Timestamp(start)) &
        (df_rfm['InvoiceDate'] <= pd.Timestamp(end))
    ].copy()
    if q_df.empty:
        continue
    q_rfm = q_df.groupby('Customer ID').agg(
        Last_Purchase=('InvoiceDate', 'max'),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    q_rfm['Recency']  = (ref - q_rfm['Last_Purchase']).dt.days
    q_rfm['Quarter']  = qname
    q_rfm['R_Score']  = pd.qcut(q_rfm['Recency'],  5, labels=[5,4,3,2,1], duplicates='drop')
    q_rfm['F_Score']  = pd.qcut(q_rfm['Frequency'].rank(method='first'),
                                 5, labels=[1,2,3,4,5], duplicates='drop')
    q_rfm['M_Score']  = pd.qcut(q_rfm['Monetary'].rank(method='first'),
                                 5, labels=[1,2,3,4,5], duplicates='drop')
    q_rfm['RFM_Score']= (
        q_rfm['R_Score'].astype(int) +
        q_rfm['F_Score'].astype(int) +
        q_rfm['M_Score'].astype(int)
    )
    quarterly_rfm[qname] = q_rfm

# Track customer score evolution across quarters
all_q = pd.concat(quarterly_rfm.values(), ignore_index=True)
pivot_scores = all_q.pivot_table(
    index='Customer ID', columns='Quarter', values='RFM_Score', aggfunc='mean'
)
pivot_scores = pivot_scores.dropna()

# Customers with consistently improving scores
pivot_scores['Q1'] = pivot_scores.get('Q1_2010', np.nan)
pivot_scores['Q4'] = pivot_scores.get('Q4_2010', np.nan)
pivot_scores['Score_Change'] = pivot_scores['Q4'] - pivot_scores['Q1']
improving   = pivot_scores[pivot_scores['Score_Change'] > 2].sort_values('Score_Change', ascending=False)
declining   = pivot_scores[pivot_scores['Score_Change'] < -2].sort_values('Score_Change')
print(f"\nCustomers with improving RFM score (Q1→Q4): {len(improving)}")
print(f"Customers with declining RFM score (Q1→Q4):  {len(declining)}")
print("\nTop 10 Improving Customers:")
print(improving.head(10))

# Heatmap of average quarterly RFM
plt.figure(figsize=(10, 5))
quarterly_avg = all_q.groupby('Quarter')['RFM_Score'].mean()
quarterly_avg.plot(kind='bar', color='teal')
plt.title('Average RFM Score by Quarter')
plt.ylabel('Average RFM Score')
plt.xlabel('Quarter')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```
<img width="514" height="345" alt="image" src="https://github.com/user-attachments/assets/28adb5d8-4847-42f6-9844-3d7f3684badc" />

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/cda4687a-63c6-40a2-ac19-6af020d21ff7" />

Weighted RFM Scoring

``` python

print("\n" + "=" * 60)
print("SECTION 8: WEIGHTED RFM SCORE")
print("=" * 60)

rfm_score_df = rfm_full[['Customer ID', 'Recency', 'Frequency', 'Monetary']].copy()

# Assign 1-5 quintile scores
rfm_score_df['R_Score'] = pd.qcut(
    rfm_score_df['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
rfm_score_df['F_Score'] = pd.qcut(
    rfm_score_df['Frequency'].rank(method='first'), 5,
    labels=[1,2,3,4,5], duplicates='drop').astype(int)
rfm_score_df['M_Score'] = pd.qcut(
    rfm_score_df['Monetary'].rank(method='first'), 5,
    labels=[1,2,3,4,5], duplicates='drop').astype(int)

# Justify weights using correlation with total revenue
correlations = rfm_score_df[['R_Score','F_Score','M_Score','Monetary']].corr()
print("\nCorrelation with Monetary Value (justifies weights):")
print(correlations['Monetary'])

# Apply weights: R=0.3, F=0.3, M=0.4
W_R, W_F, W_M = 0.3, 0.3, 0.4
rfm_score_df['Weighted_RFM'] = (
    rfm_score_df['R_Score'] * W_R +
    rfm_score_df['F_Score'] * W_F +
    rfm_score_df['M_Score'] * W_M
)

# Map to business segments
def assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    w = row['Weighted_RFM']
    if   r >= 4 and f >= 4 and m >= 4: return 'Champions'
    elif r >= 3 and f >= 3:             return 'Loyal_Customers'
    elif r >= 4 and f <= 2:             return 'Recent_Customers'
    elif r >= 3 and m >= 3:             return 'Potential_Loyalists'
    elif r == 5:                        return 'New_Customers'
    elif r <= 2 and f >= 3:             return 'At_Risk'
    elif r <= 2 and f <= 2 and m >= 3:  return 'Cannot_Lose_Them'
    elif r <= 2 and f <= 2:             return 'Lost'
    else:                               return 'Hibernating'

rfm_score_df['Segment'] = rfm_score_df.apply(assign_segment, axis=1)

segment_summary = rfm_score_df.groupby('Segment').agg(
    Count=('Customer ID', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean'),
    Avg_Weighted_RFM=('Weighted_RFM', 'mean')
).round(2)
print("\nSegment Summary:")
print(segment_summary)

# Visualize segment sizes
plt.figure(figsize=(12, 5))
segment_summary['Count'].sort_values().plot(kind='barh', color='steelblue')
plt.title('Customer Count per RFM Segment')
plt.xlabel('Number of Customers')
plt.tight_layout()
plt.show()
```

<img width="485" height="380" alt="image" src="https://github.com/user-attachments/assets/6535e1fc-1b59-46a1-842b-55e0478ac4e9" /><img width="261" height="168" alt="image" src="https://github.com/user-attachments/assets/2155d5c3-3015-42ff-98f2-c339753b4533" />

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/3ad7d787-d2db-4b6c-8dae-aea9503bc501" />

C. Advanced Clustering 

Clustering Algorithms Comparison

``` python
print("\n" + "=" * 60)
print("SECTION 9: CLUSTERING ALGORITHM COMPARISON")
print("=" * 60)

# Use base RFM scaled features for fair comparison
X_cluster = StandardScaler().fit_transform(
    rfm_full[['Recency', 'Frequency', 'Monetary']]
)
# Subsample for speed if large
if len(X_cluster) > 5000:
    idx   = np.random.choice(len(X_cluster), 5000, replace=False)
    X_sub = X_cluster[idx]
else:
    X_sub = X_cluster

results = {}

# K-Means 
t0 = time.time()
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_sub)
t_km = time.time() - t0
results['K-Means'] = {
    'labels': km.labels_,
    'time': t_km,
    'silhouette': silhouette_score(X_sub, km.labels_),
    'davies_bouldin': davies_bouldin_score(X_sub, km.labels_),
    'calinski': calinski_harabasz_score(X_sub, km.labels_)
}

# DBSCAN 
t0 = time.time()
db = DBSCAN(eps=0.5, min_samples=10).fit(X_sub)
t_db = time.time() - t0
db_labels = db.labels_
# DBSCAN may produce -1 (noise) labels; skip silhouette if only 1 cluster
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
if n_clusters_db > 1:
    mask = db_labels != -1
    results['DBSCAN'] = {
        'labels': db_labels,
        'time': t_db,
        'silhouette': silhouette_score(X_sub[mask], db_labels[mask]),
        'davies_bouldin': davies_bouldin_score(X_sub[mask], db_labels[mask]),
        'calinski': calinski_harabasz_score(X_sub[mask], db_labels[mask])
    }
    print(f"DBSCAN found {n_clusters_db} clusters, {(db_labels==-1).sum()} noise points")
else:
    print(f"DBSCAN found only {n_clusters_db} cluster — tune eps/min_samples")

# Agglomerative Hierarchical 
t0 = time.time()
agg = AgglomerativeClustering(n_clusters=4).fit(X_sub)
t_agg = time.time() - t0
results['Hierarchical'] = {
    'labels': agg.labels_,
    'time': t_agg,
    'silhouette': silhouette_score(X_sub, agg.labels_),
    'davies_bouldin': davies_bouldin_score(X_sub, agg.labels_),
    'calinski': calinski_harabasz_score(X_sub, agg.labels_)
}

# Gaussian Mixture Model 
t0 = time.time()
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit(X_sub)
gmm_labels = gmm.predict(X_sub)
t_gmm = time.time() - t0
results['GMM'] = {
    'labels': gmm_labels,
    'time': t_gmm,
    'silhouette': silhouette_score(X_sub, gmm_labels),
    'davies_bouldin': davies_bouldin_score(X_sub, gmm_labels),
    'calinski': calinski_harabasz_score(X_sub, gmm_labels)
}

# Comparison Table 
comparison_df = pd.DataFrame({
    algo: {
        'Silhouette (↑ better)': round(v['silhouette'], 4),
        'Davies-Bouldin (↓ better)': round(v['davies_bouldin'], 4),
        'Calinski-Harabasz (↑ better)': round(v['calinski'], 1),
        'Time (seconds)': round(v['time'], 3)
    }
    for algo, v in results.items()
}).T

print("\nClustering Algorithm Comparison:")
print(comparison_df.to_string())

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
metrics = ['Silhouette (↑ better)', 'Davies-Bouldin (↓ better)', 'Calinski-Harabasz (↑ better)']
for i, metric in enumerate(metrics):
    comparison_df[metric].plot(kind='bar', ax=axes[i], color='teal')
    axes[i].set_title(metric)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
plt.suptitle('Clustering Algorithm Comparison', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
```

<img width="800" height="176" alt="image" src="https://github.com/user-attachments/assets/ec82325e-9a53-43a4-a9d9-e90e91cd0776" />

<img width="1589" height="412" alt="image" src="https://github.com/user-attachments/assets/08669a8f-ab7b-4255-b845-36b8ac1ee861" />

Clustering Stability Analysis

``` python
print("\n" + "=" * 60)
print("SECTION 10: CLUSTER STABILITY ANALYSIS (50 runs)")
print("=" * 60)

stability_results = {}
for k in range(3, 9):
    ari_scores = []
    labels_list = []
    for seed in range(50):
        km_s = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(X_sub)
        labels_list.append(km_s.labels_)
    # Compare each run against the first run using Adjusted Rand Index
    for i in range(1, len(labels_list)):
        ari = adjusted_rand_score(labels_list[0], labels_list[i])
        ari_scores.append(ari)
    stability_results[k] = {
        'mean_ARI': np.mean(ari_scores),
        'std_ARI':  np.std(ari_scores)
    }
    print(f"K={k}: Mean ARI={np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")

stability_df = pd.DataFrame(stability_results).T
plt.figure(figsize=(9, 4))
plt.errorbar(stability_df.index, stability_df['mean_ARI'],
             yerr=stability_df['std_ARI'], marker='o', capsize=5, color='steelblue')
plt.xlabel('Number of Clusters K')
plt.ylabel('Adjusted Rand Index (higher = more stable)')
plt.title('Cluster Stability Analysis (50 random seeds)')
plt.xticks(stability_df.index)
plt.tight_layout()
plt.show()

best_k = stability_df['mean_ARI'].idxmax()
print(f"\nMost stable K: {best_k} (ARI={stability_df.loc[best_k,'mean_ARI']:.4f})")
```

<img width="422" height="169" alt="image" src="https://github.com/user-attachments/assets/53fa41fe-b04a-4249-902c-38ec42da34b4" />

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/e60ea77a-fe1f-445f-b9e7-4adac240bd23" />

<img width="210" height="38" alt="image" src="https://github.com/user-attachments/assets/f5714256-eb5a-434a-833d-f4a9e5ba11ae" />

PCA and UMAP visualization

``` python
print("\n" + "=" * 60)
print("SECTION 11: PCA vs UMAP CLUSTER VISUALIZATION")
print("=" * 60)

# Use best_k clusters
km_final = KMeans(n_clusters=int(best_k), random_state=42, n_init=10).fit(X_sub)
labels_final = km_final.labels_

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sub)

fig, axes = plt.subplots(1, 2 if not UMAP_AVAILABLE else 2, figsize=(14, 5))
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                          c=labels_final, cmap='tab10', s=5, alpha=0.6)
axes[0].set_title(f'PCA Projection (K={best_k} clusters)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[0])

# UMAP (if available)
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap  = reducer.fit_transform(X_sub)
    scatter2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1],
                               c=labels_final, cmap='tab10', s=5, alpha=0.6)
    axes[1].set_title(f'UMAP Projection (K={best_k} clusters)')
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    plt.colorbar(scatter2, ax=axes[1])
else:
    # 3D PCA as fallback
    pca3 = PCA(n_components=3)
    X_3d = pca3.fit_transform(X_sub)
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2],
                 c=labels_final, cmap='tab10', s=3, alpha=0.5)
    ax3d.set_title('3D PCA Projection')

plt.tight_layout()
plt.show()
```

<img width="434" height="66" alt="image" src="https://github.com/user-attachments/assets/5503f6ec-67c1-425e-934f-6cec340e106d" />

<img width="1356" height="490" alt="image" src="https://github.com/user-attachments/assets/e7d635b5-9293-44cd-8398-ff9f4383f255" />

D. Segment Analysis & Business Intelligence

Pareto Analysis

``` python
print("\n" + "=" * 60)
print("SECTION 12: PARETO ANALYSIS — REVENUE CONCENTRATION")
print("=" * 60)

# Merge segment labels with RFM
rfm_full_seg = rfm_full.merge(
    rfm_score_df[['Customer ID', 'Segment', 'Weighted_RFM']],
    on='Customer ID', how='left'
)

# Sort by monetary descending
pareto = rfm_full_seg[['Customer ID','Monetary','Segment']].sort_values(
    'Monetary', ascending=False
).reset_index(drop=True)
pareto['Cumulative_Revenue']     = pareto['Monetary'].cumsum()
pareto['Cumulative_Revenue_Pct'] = pareto['Cumulative_Revenue'] / pareto['Monetary'].sum() * 100
pareto['Customer_Pct']           = (pareto.index + 1) / len(pareto) * 100

# Find 80% revenue threshold
pareto_80 = pareto[pareto['Cumulative_Revenue_Pct'] <= 80]
pct_customers_for_80_pct = pareto_80['Customer_Pct'].max()
print(f"\nPareto Principle: {pct_customers_for_80_pct:.1f}% of customers generate 80% of revenue")
print(f"Total customers: {len(pareto):,}")
print(f"Top customers driving 80% revenue: {len(pareto_80):,}")

# Which segments make up the top 20%?
top_20_cutoff = int(len(pareto) * 0.2)
top_20_seg = pareto.iloc[:top_20_cutoff]['Segment'].value_counts()
print("\nSegment composition of top 20% revenue customers:")
print(top_20_seg)

# Plot Pareto curve
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(pareto['Customer_Pct'], pareto['Monetary'], width=0.1,
        color='steelblue', alpha=0.5, label='Individual Revenue')
ax2 = ax1.twinx()
ax2.plot(pareto['Customer_Pct'], pareto['Cumulative_Revenue_Pct'],
         'r-', linewidth=2, label='Cumulative Revenue %')
ax2.axhline(80, color='green', linestyle='--', label='80% threshold')
ax2.axvline(pct_customers_for_80_pct, color='orange', linestyle='--',
            label=f'{pct_customers_for_80_pct:.1f}% customers')
ax1.set_xlabel('Customer Percentile (%)')
ax1.set_ylabel('Revenue (£)', color='steelblue')
ax2.set_ylabel('Cumulative Revenue (%)', color='red')
ax2.legend(loc='center right')
plt.title('Pareto Revenue Concentration Analysis')
plt.tight_layout()
plt.savefig('output/pareto_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

<img width="422" height="288" alt="image" src="https://github.com/user-attachments/assets/e3385092-3e81-4576-ba4a-fd77c654a214" />

<img width="1089" height="490" alt="image" src="https://github.com/user-attachments/assets/db7de294-7d66-4dfa-bbfd-0dc9686589a6" />

Segment Migration Matrix

``` python
print("\n" + "=" * 60)
print("SECTION 13: SEGMENT MIGRATION MATRIX (Q1 → Q4 2010)")
print("=" * 60)

def get_quarterly_segments(q_df, ref_date, quarter_name):
    """Calculate RFM segments for a specific quarter."""
    rfm = q_df.groupby('Customer ID').agg(
        Last_Purchase=('InvoiceDate', 'max'),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    rfm['Recency'] = (ref_date - rfm['Last_Purchase']).dt.days

    rfm['R'] = pd.qcut(rfm['Recency'],  3, labels=[3,2,1], duplicates='drop').astype(int)
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'),
                        3, labels=[1,2,3], duplicates='drop').astype(int)
    rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'),
                        3, labels=[1,2,3], duplicates='drop').astype(int)
    rfm['RFM'] = rfm['R'] + rfm['F'] + rfm['M']

    def seg(score):
        if score >= 8: return 'Champions'
        elif score >= 6: return 'Loyal'
        elif score >= 4: return 'At_Risk'
        else: return 'Lost'

    rfm[f'Segment_{quarter_name}'] = rfm['RFM'].apply(seg)
    return rfm[['Customer ID', f'Segment_{quarter_name}']]

q1_data = df_rfm[
    (df_rfm['InvoiceDate'] >= '2010-01-01') &
    (df_rfm['InvoiceDate'] <= '2010-03-31')
]
q4_data = df_rfm[
    (df_rfm['InvoiceDate'] >= '2010-10-01') &
    (df_rfm['InvoiceDate'] <= '2010-12-31')
]

if not q1_data.empty and not q4_data.empty:
    seg_q1 = get_quarterly_segments(q1_data, pd.Timestamp('2010-04-01'), 'Q1')
    seg_q4 = get_quarterly_segments(q4_data, pd.Timestamp('2011-01-01'), 'Q4')

    migration = seg_q1.merge(seg_q4, on='Customer ID', how='inner')
    migration_matrix = pd.crosstab(
        migration['Segment_Q1'],
        migration['Segment_Q4'],
        normalize='index'
    ).round(3) * 100

    print("\nSegment Migration Matrix (Q1 2010 → Q4 2010, % of Q1 segment):")
    print(migration_matrix)

    plt.figure(figsize=(9, 6))
    sns.heatmap(migration_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': '% of Q1 Customers'})
    plt.title('Customer Segment Migration Matrix (Q1 → Q4 2010)')
    plt.ylabel('Q1 2010 Segment')
    plt.xlabel('Q4 2010 Segment')
    plt.tight_layout()
    plt.show()

    # Retention rate per segment
    for seg in migration_matrix.index:
        if seg in migration_matrix.columns:
            retention = migration_matrix.loc[seg, seg]
            print(f"{seg} retention rate: {retention:.1f}%")
else:
    print("Insufficient quarterly data — check date ranges in dataset")
```

<img width="476" height="204" alt="image" src="https://github.com/user-attachments/assets/cb5dd8fd-96ad-490d-ab2e-590679f89277" />

<img width="838" height="590" alt="image" src="https://github.com/user-attachments/assets/48c543a4-a76a-446b-9c0e-683bebbb5b9c" />

<img width="227" height="84" alt="image" src="https://github.com/user-attachments/assets/e2471e94-afda-4ed5-9761-1d9ad2cec70c" />

Survival Analysis

``` python
print("\n" + "=" * 60)
print("SECTION 14: SURVIVAL ANALYSIS — CUSTOMER CHURN TIMING")
print("=" * 60)

if LIFELINES_AVAILABLE:
    # Build survival dataset: days active before going dormant
    customer_dates = df_rfm.groupby('Customer ID').agg(
        First_Purchase=('InvoiceDate', 'min'),
        Last_Purchase=('InvoiceDate', 'max'),
    ).reset_index()
    customer_dates['Duration_Days'] = (
        customer_dates['Last_Purchase'] - customer_dates['First_Purchase']
    ).dt.days
    # Observed churn = last purchase > 90 days before reference date
    customer_dates['Churned'] = (
        (REFERENCE_DATE - customer_dates['Last_Purchase']).dt.days > 90
    ).astype(int)
    customer_dates = customer_dates[customer_dates['Duration_Days'] >= 0]

    kmf = KaplanMeierFitter()
    kmf.fit(
        customer_dates['Duration_Days'],
        event_observed=customer_dates['Churned'],
        label='All Customers'
    )

    plt.figure(figsize=(11, 5))
    kmf.plot_survival_function()
    plt.axhline(0.5, color='red', linestyle='--', label='50% Survival')
    median_survival = kmf.median_survival_time_
    plt.axvline(median_survival, color='orange', linestyle='--',
                label=f'Median = {median_survival:.0f} days')
    plt.xlabel('Days Since First Purchase')
    plt.ylabel('Survival Probability (Customer Still Active)')
    plt.title('Kaplan-Meier Customer Survival Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/survival_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nMedian customer survival time: {median_survival:.0f} days")
    print("Interpretation: 50% of customers go dormant within this many days")
    print(f"Recommendation: Trigger win-back campaign at day {int(median_survival * 0.7)}")
else:
    print("lifelines not installed — run: pip install lifelines")
```

<img width="421" height="94" alt="image" src="https://github.com/user-attachments/assets/3410f8ec-eed9-473a-8b21-eaa3e5570bf9" />

One Time Buyer Prediction

``` python

print("\n" + "=" * 60)
print("SECTION 15: ONE-TIME BUYER → REPEAT BUYER PREDICTION")
print("=" * 60)

# Identify one-time buyers
customer_invoice_counts = df_rfm.groupby('Customer ID')['Invoice'].nunique()
one_time_buyers = customer_invoice_counts[customer_invoice_counts == 1].index

# Build features from their first (only) purchase
first_purchases = df_rfm.sort_values('InvoiceDate').drop_duplicates('Customer ID')
one_time_df = first_purchases[first_purchases['Customer ID'].isin(one_time_buyers)].copy()

# Target: did they return within 90 days?
# For training purposes, use customers with first purchase early enough
first_purchase_dates = df_rfm.groupby('Customer ID')['InvoiceDate'].min().reset_index()
first_purchase_dates.columns = ['Customer ID', 'First_Purchase_Date']

all_customer_counts = customer_invoice_counts.reset_index()
all_customer_counts.columns = ['Customer ID', 'Invoice_Count']

predict_df = first_purchase_dates.merge(all_customer_counts, on='Customer ID')
predict_df['Has_Returned'] = (predict_df['Invoice_Count'] > 1).astype(int)

# Features from first purchase
first_purch_features = df_rfm.sort_values('InvoiceDate').groupby('Customer ID').first().reset_index()
predict_df = predict_df.merge(
    first_purch_features[['Customer ID', 'Price', 'Quantity', 'DayOfWeek', 'Hour', 'Quarter']],
    on='Customer ID', how='left'
)
predict_df['Revenue'] = predict_df['Price'] * predict_df['Quantity']
predict_df = predict_df.dropna()

feature_cols = ['Price', 'Quantity', 'Revenue', 'DayOfWeek', 'Hour', 'Quarter']
X_pred = predict_df[feature_cols]
y_pred = predict_df['Has_Returned']

X_tr, X_te, y_tr, y_te = train_test_split(X_pred, y_pred, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_tr, y_tr)
lr_score = lr.score(X_te, y_te)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_tr, y_tr)
rf_score = rf.score(X_te, y_te)

print(f"\nLogistic Regression Accuracy: {lr_score:.3f}")
print(f"Random Forest Accuracy:       {rf_score:.3f}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_te, rf.predict(X_te),
      target_names=['One-Time', 'Repeat Buyer']))

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importance for Repeat Purchase Prediction:")
print(feat_imp)

plt.figure(figsize=(8, 4))
feat_imp.plot(kind='bar', color='teal')
plt.title('Feature Importance: One-Time → Repeat Buyer Prediction')
plt.ylabel('Importance')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
```
<img width="413" height="446" alt="image" src="https://github.com/user-attachments/assets/443a241e-71bd-4f63-90c7-6a797ec94dfd" />

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/d1a69763-35ec-451a-a1a7-a27450d57f4f" />


E. Customer Lifetime Value Prediction

Customer lifetime Value

``` python

print("\n" + "=" * 60)
print("SECTION 16: CUSTOMER LIFETIME VALUE PREDICTION")
print("=" * 60)

if LIFELINES_AVAILABLE:
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data

        # Prepare transaction summary for BG/NBD
        df_rfm_clv = df_rfm.copy()
        df_rfm_clv['InvoiceDate'] = pd.to_datetime(df_rfm_clv['InvoiceDate'])

        summary = summary_data_from_transaction_data(
            df_rfm_clv,
            customer_id_col='Customer ID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            observation_period_end=REFERENCE_DATE
        )
        summary = summary[summary['frequency'] > 0]

        # BG/NBD Model
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])
        summary['Expected_Purchases_90d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            90, summary['frequency'], summary['recency'], summary['T']
        )

        # Gamma-Gamma Model
        returning_customers = summary[summary['frequency'] > 0]
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])

        summary['Expected_Revenue_Per_Purchase'] = ggf.conditional_expected_average_profit(
            returning_customers['frequency'],
            returning_customers['monetary_value']
        )

        # 12-month CLV
        summary['CLV_12_Months'] = ggf.customer_lifetime_value(
            bgf,
            returning_customers['frequency'],
            returning_customers['recency'],
            returning_customers['T'],
            returning_customers['monetary_value'],
            time=12,
            discount_rate=0.01
        )

        print("\nTop 20 Customers by Predicted 12-Month CLV:")
        print(summary.sort_values('CLV_12_Months', ascending=False).head(20))

        plt.figure(figsize=(10, 5))
        summary['CLV_12_Months'].hist(bins=50, color='steelblue', edgecolor='white')
        plt.xlabel('Predicted 12-Month CLV (£)')
        plt.ylabel('Number of Customers')
        plt.title('Distribution of Predicted Customer Lifetime Value')
        plt.tight_layout()
        plt.savefig('output/clv_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

    except ImportError:
        print("Run: pip install lifetimes")
else:
    print("Install lifelines/lifetimes: pip install lifetimes")
```
<img width="448" height="98" alt="image" src="https://github.com/user-attachments/assets/ed31b87f-843d-45f6-ab0a-1414413da876" />

Premium Signal Products

``` python

print("\n" + "=" * 60)
print("SECTION 18: PREMIUM SIGNAL PRODUCTS")
print("=" * 60)

rfm_seg_merged = df_clean.merge(
    rfm_score_df[['Customer ID', 'Segment']], on='Customer ID', how='left'
)

champions_products = set(
    rfm_seg_merged[rfm_seg_merged['Segment'] == 'Champions']['Description'].dropna().unique()
)
lost_products = set(
    rfm_seg_merged[rfm_seg_merged['Segment'] == 'Lost']['Description'].dropna().unique()
)

exclusive_champion_products = champions_products - lost_products
print(f"\nProducts bought by Champions but NEVER by Lost customers: {len(exclusive_champion_products)}")

# Rank by purchase frequency among Champions
champion_product_freq = (
    rfm_seg_merged[
        (rfm_seg_merged['Segment'] == 'Champions') &
        (rfm_seg_merged['Description'].isin(exclusive_champion_products))
    ]
    .groupby('Description')['Invoice'].count()
    .sort_values(ascending=False)
    .head(20)
)
print("\nTop 20 Premium Signal Products:")
print(champion_product_freq)

plt.figure(figsize=(12, 6))
champion_product_freq.plot(kind='barh', color='gold')
plt.title('Top 20 Premium Signal Products\n(Bought by Champions, Never by Lost Customers)')
plt.xlabel('Purchase Frequency Among Champions')
plt.tight_layout()
plt.show()
```

<img width="437" height="496" alt="image" src="https://github.com/user-attachments/assets/7a1316d1-dad5-4d7f-bdda-7a67d1f453dc" />

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/a57b2535-fda1-4b2f-9909-7c5930348a1b" />

3. Weighted RFM ScoringA weighted score was implemented: $0.3 \times R + 0.3 \times F + 0.4 \times M$.Justification:

Correlation analysis showed Frequency and Monetary have a log-correlation of 0.85, making them the strongest predictors of customer value. Recency ($\rho = -0.50$) is critical for churn detection.

6. Clustering ComparisonWe compared four algorithms on the RFM space:

| Algorithm | Silhouette Score | Davies-Bouldin | Time (s) |
| -------- | -------- | -------- | -------- | 
| K-Means | 0.589 | 0.586 | 0.81 |
| Agglomerative | 0.590 | 0.628 | 1.71 |
| GMM | 0.183 | 1.189 | 0.95 |

Conclusion: While Agglomerative clustering had a slightly higher silhouette, K-Means was chosen for its computational efficiency and ease of business interpretability.

Hypothesis Testing

``` python

print("\n" + "=" * 60)
print("SECTION 19: HYPOTHESIS TESTING")
print("=" * 60)

# FIX: Ensure rfm_full_seg is defined by merging base features with segments
# This step is likely what was missing in your active memory/cell
rfm_full_seg = rfm_full.merge(
    rfm_score_df[['Customer ID', 'Segment', 'Weighted_RFM']],
    on='Customer ID', 
    how='left'
)

# Drop any rows where segmenting might have failed
rfm_ht = rfm_full_seg.dropna(subset=['Segment'])

# ── Q20: Mann-Whitney U — Champions vs At-Risk Monetary ─────
print("\n── Test 1: Mann-Whitney U — Champions vs At-Risk Monetary Value ──")
champions_mon = rfm_ht[rfm_ht['Segment'] == 'Champions']['Monetary']
at_risk_mon   = rfm_ht[rfm_ht['Segment'] == 'At_Risk']['Monetary']

if len(champions_mon) > 0 and len(at_risk_mon) > 0:
    stat, p = stats.mannwhitneyu(champions_mon, at_risk_mon, alternative='greater')
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt((champions_mon.std()**2 + at_risk_mon.std()**2) / 2)
    cohens_d = (champions_mon.mean() - at_risk_mon.mean()) / pooled_std

    print(f"Mann-Whitney U Statistic: {stat:.2f}")
    print(f"P-value: {p:.6f}")
    print(f"Cohen's d (effect size): {cohens_d:.4f}")
    
    if p < 0.05:
        print("✅ REJECT H0: Champions have significantly higher monetary value than At-Risk")
    else:
        print("❌ FAIL TO REJECT H0")
else:
    print("Insufficient data for one or both segments")

# ── Q21: Kruskal-Wallis — Country RFM Differences ───────────
print("\n── Test 2: Kruskal-Wallis — Country RFM Profile Differences ──")
# We need to map countries back to the RFM table
country_map = df_clean[['Customer ID', 'Country']].drop_duplicates()
country_rfm = rfm_score_df.merge(country_map, on='Customer ID', how='left')

top_countries = country_rfm['Country'].value_counts().head(8).index
country_groups = [
    country_rfm[country_rfm['Country'] == c]['M_Score'].dropna()
    for c in top_countries
]

stat_kw, p_kw = stats.kruskal(*country_groups)
print(f"Kruskal-Wallis H Statistic: {stat_kw:.4f}")
print(f"P-value: {p_kw:.6f}")

if p_kw < 0.05:
    print("✅ REJECT H0: Country significantly affects RFM profile")
else:
    print("❌ FAIL TO REJECT H0")

# ── Q22: Chi-Square — Seasonal Purchase → Segment ───────────
print("\n── Test 3: Chi-Square — Seasonal Timing → Segment Assignment ──")
# Get the first quarter each customer appeared in
first_q = df_rfm.sort_values('InvoiceDate').groupby('Customer ID')['Quarter'].first().reset_index()
first_q.columns = ['Customer ID', 'First_Quarter']

rfm_season = rfm_ht.merge(first_q, on='Customer ID', how='left')

contingency = pd.crosstab(rfm_season['First_Quarter'], rfm_season['Segment'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_chi:.6f}")

if p_chi < 0.05:
    print("✅ REJECT H0: Season of first purchase significantly predicts long-term segment")
else:
    print("❌ FAIL TO REJECT H0")
```

<img width="612" height="319" alt="image" src="https://github.com/user-attachments/assets/b8862822-2c68-4848-9f39-86a4d5e79437" />

Insights:

1. Champions vs. At-Risk (Monetary Value):

- Mann-Whitney U Test (due to non-normal distribution): $p < 0.001$, Cohen’s $d = 0.563$.Insight: There is a statistically significant difference in spend between these segments, confirming that our segmentation logic effectively separates high-value and low-value groups. 

2. Country Distribution:

- Kruskal-Wallis: Significant differences found ($p < 0.05$). The UK dominates in volume, but countries like EIRE and Netherlands show higher average spend per customer.

Business Action Summary

``` python

print("\n" + "=" * 60)
print("SECTION 20: BUSINESS ACTION SUMMARY BY SEGMENT")
print("=" * 60)

action_plan = {
    'Champions': {
        'Strategy': 'Reward & Retain',
        'Action': 'VIP loyalty program, early access to new products, referral incentives',
        'Metric': 'Maintain >85% retention rate'
    },
    'Loyal_Customers': {
        'Strategy': 'Upsell & Cross-sell',
        'Action': 'Premium product recommendations, bundle offers',
        'Metric': 'Increase AOV by 15%'
    },
    'At_Risk': {
        'Strategy': 'Win-Back Campaign',
        'Action': f'Trigger email at day {int(median_survival * 0.7) if LIFELINES_AVAILABLE else 45} inactivity with 10% discount',
        'Metric': 'Recover 25% of At-Risk customers'
    },
    'Lost': {
        'Strategy': 'Re-engagement or Deprioritize',
        'Action': 'One final win-back offer; if no response, remove from active campaigns',
        'Metric': 'Cost savings from reduced marketing spend'
    },
    'New_Customers': {
        'Strategy': 'Onboarding & Education',
        'Action': 'Welcome series, product education, second purchase incentive within 30 days',
        'Metric': 'Convert 40% to Loyal within 90 days'
    },
    'Potential_Loyalists': {
        'Strategy': 'Nurture to Loyalty',
        'Action': 'Personalized recommendations based on purchase history, loyalty points',
        'Metric': 'Move 30% to Loyal segment within 6 months'
    }
}

for segment, plan in action_plan.items():
    seg_count = rfm_score_df[rfm_score_df['Segment'] == segment]['Customer ID'].count()
    seg_revenue = rfm_ht[rfm_ht['Segment'] == segment]['Monetary'].sum()
    print(f"\n{'─'*50}")
    print(f"Segment: {segment} ({seg_count} customers | £{seg_revenue:,.0f} revenue)")
    for k, v in plan.items():
        print(f"  {k}: {v}")

# Save final RFM dataset
os.makedirs('output', exist_ok=True)
rfm_full_seg.to_csv('output/final_rfm_segmentation.csv', index=False)
rfm_score_df.to_csv('output/rfm_scored_segments.csv', index=False)
print("\n\nAll outputs saved to /output/ directory")
print("Files: final_rfm_segmentation.csv, rfm_scored_segments.csv")
```
<img width="553" height="459" alt="image" src="https://github.com/user-attachments/assets/854cc59b-01b8-42a7-845e-c3296120b69f" />
<img width="581" height="246" alt="image" src="https://github.com/user-attachments/assets/f44e99cf-b018-4329-b5ad-8b665b74b5bc" />


**SQL**
1). Data Quality Audit

``` sql
SELECT
CASE
  WHEN STARTS_WITH(Invoice, 'C') THEN 'Cancellation'
  WHEN Price < 0                 THEN 'Negative_Price_Adjustment'
  WHEN Price = 0 AND Quantity > 0 THEN 'Free_Sample_Promotional'
  WHEN Quantity < 0 AND NOT STARTS_WITH(Invoice, 'C') THEN 'Return_Without_C_Flag'
  ELSE 'Valid_Purchase'
  END AS Row_Type,
  COUNT(*) AS Row_Count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS Row_Pct,
  ROUND(SUM(Quantity * Price), 2) AS Total_Revenue,
  ROUND(AVG(Price), 2) AS Avg_Price,
  ROUND(AVG(Quantity), 2) AS Avg_Quantity
FROM
    `customersegment.retail.segment`
GROUP BY
    Row_Type
ORDER BY
    Row_Count DESC;
```

<img width="1100" height="308" alt="image" src="https://github.com/user-attachments/assets/5e3393d2-a39d-416f-869c-4ad8c51dd842" />

2). Missing Customer ID PAttern Analysis

``` sql
SELECT
CASE WHEN customer_id IS NULL THEN 'Missing_ID' ELSE 'Has_ID' END AS ID_Status,
COUNT(*) AS Transaction_Count,
ROUND(AVG(Quantity * Price), 2)              AS Avg_Order_Value,
ROUND(AVG(Quantity), 2)                      AS Avg_Quantity,
COUNT(DISTINCT Country)                      AS Unique_Countries,
ROUND(AVG(EXTRACT(HOUR FROM InvoiceDate)), 1)AS Avg_Purchase_Hour,
COUNT(DISTINCT Country)                      AS Country_Spread
FROM
    `customersegment.retail.segment`
GROUP BY
ID_Status;

-- Top countries with missing Customer IDs
SELECT
    Country,
    COUNT(*) AS Missing_ID_Transactions,
    ROUND(AVG(Quantity * Price), 2) AS Avg_Revenue
FROM
    `customersegment.retail.segment`
WHERE
    customer_id IS NULL
GROUP BY
    Country
ORDER BY
    Missing_ID_Transactions DESC
LIMIT 15;
```

<img width="1258" height="154" alt="image" src="https://github.com/user-attachments/assets/8cc541d3-5b23-4ad5-b4b5-e0fc7418aaba" />

3). Full RFM Calulations
``` sql

WITH clean_data AS (
    -- Step 1: Filter to valid purchases only
    SELECT
        CAST(customer_id AS INT64) AS CustomerID,
        Invoice,
        InvoiceDate,
        Quantity * Price                         AS LineRevenue
    FROM
        `customersegment.retail.segment`
    WHERE
        customer_id IS NOT NULL
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
    DAY) AS Recency_Days,
    COUNT(DISTINCT Invoice) AS Frequency,
    ROUND(SUM(LineRevenue), 2) AS Monetary
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
        6 - NTILE(5) OVER (ORDER BY Recency_Days ASC) AS R_Score,
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
        ROUND(R_Score * 0.3 + F_Score * 0.3 + M_Score * 0.4, 2) AS Weighted_RFM,
        CASE
          WHEN R_Score >= 4 AND F_Score >= 4 AND M_Score >= 4 THEN 'Champions'
          WHEN R_Score >= 3 AND F_Score >= 3   THEN 'Loyal_Customers'
          WHEN R_Score >= 4 AND F_Score <= 2   THEN 'Recent_Customers'
          WHEN R_Score >= 3 AND M_Score >= 3   THEN 'Potential_Loyalists'
          WHEN R_Score = 5                     THEN 'New_Customers'
          WHEN R_Score <= 2 AND F_Score >= 3   THEN 'At_Risk'
          WHEN R_Score <= 2 AND F_Score <= 2 AND M_Score >= 3 THEN 'Cannot_Lose_Them'
          WHEN R_Score <= 2 AND F_Score <= 2   THEN 'Lost'
          ELSE 'Hibernating'
        END AS Segment
    FROM
        rfm_scored
)

SELECT *
FROM rfm_combined
ORDER BY Monetary DESC;
```
<img width="1614" height="485" alt="image" src="https://github.com/user-attachments/assets/ba2452ab-d1cf-40a2-89fc-6f6bd23203d6" />

4). PARETO ANALYSIS - Cummulatiev Revenue
``` sql
WITH customer_revenue AS (
    SELECT
        CAST(customer_id AS INT64) AS CustomerID,
        ROUND(SUM(Quantity * Price), 2) AS Total_Revenue
    FROM
        `customersegment.retail.segment`
    WHERE
        customer_id IS NOT NULL
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
        CAST(customer_id AS INT64)                AS CustomerID,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue
    FROM 
    `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
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
```
<img width="1015" height="575" alt="image" src="https://github.com/user-attachments/assets/db0edbe4-33d8-4129-a31e-eef2a5512503" />

5).COHORT RETENTION TABLE
``` sql

WITH clean AS (
    SELECT
        CAST(customer_id AS INT64)                AS CustomerID,
        DATE_TRUNC(DATE(InvoiceDate), MONTH)     AS Purchase_Month
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
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
WHERE r.Months_Since_First <= 11               
ORDER BY r.Cohort_Month, r.Months_Since_First;
```
<img width="899" height="561" alt="image" src="https://github.com/user-attachments/assets/9e64aff4-24ee-45c5-ad31-6ccb8d5e2e4f" />

6). Churned Customers Identification
``` sql
WITH customer_halves AS (
    SELECT
    CAST(customer_id AS INT64) AS CustomerID,
    MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
    THEN 1 ELSE 0 END) AS Active_H1,
    MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-06-01' AND '2010-12-31'
        THEN 1 ELSE 0 END) AS Active_H2,
    SUM(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
    THEN Quantity * Price ELSE 0 END) AS H1_Revenue,
    COUNT(DISTINCT CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                    THEN Invoice END)    AS H1_Frequency,
    DATE_DIFF(DATE '2010-05-31',
              DATE(MIN(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                          THEN InvoiceDate END)),
    DAY)AS H1_Recency_At_End
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY customer_id)

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
        CAST(customer_id AS INT64)                AS CustomerID,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN 1 ELSE 0 END)              AS Active_H1,
        MAX(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-06-01' AND '2010-12-31'
                 THEN 1 ELSE 0 END)              AS Active_H2,
        SUM(CASE WHEN DATE(InvoiceDate) BETWEEN '2010-01-01' AND '2010-05-31'
                 THEN Quantity * Price ELSE 0 END) AS H1_Revenue
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY customer_id
)
SELECT
    ROUND(SUM(CASE WHEN Active_H1=1 AND Active_H2=0 THEN 1 ELSE 0 END) * 100.0
          / NULLIF(SUM(Active_H1), 0), 2)        AS Churn_Rate_Pct,
    ROUND(AVG(CASE WHEN Active_H1=1 AND Active_H2=0 THEN H1_Revenue END), 2)
                                                 AS Avg_Revenue_Of_Churned,
    ROUND(AVG(CASE WHEN Active_H1=1 AND Active_H2=1 THEN H1_Revenue END), 2)
                                                 AS Avg_Revenue_Of_Retained
FROM customer_halves;
```
<img width="1131" height="472" alt="image" src="https://github.com/user-attachments/assets/1df7299b-6a4a-422f-a8ac-937cd995cb0c" />

7). Inter-purchase interval Analysis
``` sql
WITH customer_purchases AS (
    SELECT
        customer_id,
        Purchase_Date,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY Purchase_Date) AS Purchase_Number
    FROM (
        SELECT
            CAST(customer_id AS INT64) AS customer_id,
            DATE(invoicedate) AS Purchase_Date
        FROM `customersegment.retail.segment`
        WHERE customer_id IS NOT NULL AND NOT STARTS_WITH(invoice, 'C') AND quantity > 0 AND price > 0
        GROUP BY 1, 2  
    )),

with_lag AS (
    SELECT
        customer_id,
        Purchase_Date,
        Purchase_Number,
        LAG(Purchase_Date) OVER (PARTITION BY customer_id ORDER BY Purchase_Date) AS Prev_Purchase_Date,
        MAX(Purchase_Number) OVER (
            PARTITION BY customer_id
        ) AS Max_Purchase_Number
    FROM customer_purchases
),

intervals AS (
    SELECT
        customer_id,
        Purchase_Date,
        Purchase_Number,
        Max_Purchase_Number,
        DATE_DIFF(Purchase_Date, Prev_Purchase_Date, DAY) AS Days_Since_Last_Purchase FROM with_lag WHERE Prev_Purchase_Date IS NOT NULL
)

SELECT
    customer_id,
    ROUND(AVG(Days_Since_Last_Purchase), 1) AS Avg_Interval_Days,
    COUNT(*) AS Num_Intervals,
    ROUND(
        SAFE_DIVIDE(
        SUM(CASE WHEN Purchase_Number > (Max_Purchase_Number / 2) THEN Days_Since_Last_Purchase ELSE 0 END),
        NULLIF(COUNTIF(Purchase_Number > (Max_Purchase_Number / 2)), 0))
         - 
        SAFE_DIVIDE(
            SUM(CASE WHEN Purchase_Number <= (Max_Purchase_Number / 2) THEN Days_Since_Last_Purchase ELSE 0 END),
            NULLIF(COUNTIF(Purchase_Number <= (Max_Purchase_Number / 2)), 0)), 1) AS Interval_Trend_Days
FROM intervals
GROUP BY customer_id
HAVING COUNT(*) >= 2
ORDER BY Interval_Trend_Days DESC;
```
<img width="717" height="480" alt="image" src="https://github.com/user-attachments/assets/96fa0057-26e9-4ef4-a1ef-8b72a118d4b8" />

8). Product Co-occurence
``` sql

WITH valid_transactions AS (
    SELECT
        Invoice,
        TRIM(Description) AS Product
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND Description IS NOT NULL
      AND Country = 'United Kingdom'             
),

product_freq AS (
    SELECT
        Product,
        COUNT(DISTINCT Invoice) AS Product_Transactions
    FROM valid_transactions
    GROUP BY Product
),

total_transactions AS (
    SELECT COUNT(DISTINCT Invoice)  AS N
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
    AND a.Product < b.Product                   -- Avoid duplicates (A,B) vs (B,A)
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
```
<img width="577" height="468" alt="image" src="https://github.com/user-attachments/assets/b56bcb51-db07-40f8-80e0-b986f80b6c02" /><img width="1083" height="470" alt="image" src="https://github.com/user-attachments/assets/b9fbca00-7be3-452f-bc57-c30d845851fe" />

9). Revenue per Unique customer by product
``` sql
SELECT
    TRIM(Description)                            AS Product,
    COUNT(DISTINCT customer_id)                   AS Unique_Customers,
    ROUND(SUM(Quantity * Price), 2)              AS Total_Revenue,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT customer_id), 0), 2)
                                                 AS Revenue_Per_Customer,
    ROUND(AVG(Quantity * Price), 2)              AS Avg_Transaction_Value,
    COUNT(DISTINCT Invoice)                      AS Total_Purchases
FROM `customersegment.retail.segment`
WHERE customer_id IS NOT NULL
  AND NOT STARTS_WITH(Invoice, 'C')
  AND Quantity > 0 AND Price > 0
  AND Description IS NOT NULL
GROUP BY TRIM(Description)
HAVING COUNT(DISTINCT customer_id) >= 10         -- Exclude one-off items
ORDER BY Revenue_Per_Customer DESC
LIMIT 30;
```
<img width="1107" height="375" alt="image" src="https://github.com/user-attachments/assets/2977f86e-18c7-4000-8071-e018f2d45cc3" />

10). Declining product Sales Valocity
``` sql

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
    FROM `customersegment.retail.segment`
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
```
<img width="1457" height="516" alt="image" src="https://github.com/user-attachments/assets/8fca54f3-b7f9-482d-942f-b92b77969e2e" />

11). Customer Country perrcentile Analysis
``` sql
WITH customer_revenue AS (
    SELECT
    CAST(customer_id AS INT64) AS customer_id,
    Country,
    ROUND(SUM(Quantity * Price), 2) AS Total_Revenue
   FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY customer_id, Country
),

with_percentiles AS (
    SELECT
        customer_id,
        Country,
        Total_Revenue,
        PERCENT_RANK() OVER (
            PARTITION BY Country
            ORDER BY Total_Revenue ASC) AS Country_Percentile,
        PERCENT_RANK() OVER (
            ORDER BY Total_Revenue ASC) AS Global_Percentile
    FROM customer_revenue
)

SELECT
    customer_id,
    Country,
    Total_Revenue,
    ROUND(Country_Percentile * 100, 1) AS Country_Percentile_Pct,
    ROUND(Global_Percentile * 100, 1)  AS Global_Percentile_Pct,
    CASE
        WHEN Country_Percentile >= 0.90 AND Global_Percentile < 0.90
        THEN 'Local_Champion_Underserved'
        WHEN Global_Percentile >= 0.90           
        THEN 'Global_Champion'
        WHEN Country_Percentile >= 0.90          
        THEN 'Local_Champion'
        ELSE 'Standard'
    END AS Customer_Tier
FROM with_percentiles
WHERE Country_Percentile >= 0.90
ORDER BY Country, Country_Percentile DESC;
```
<img width="1191" height="428" alt="image" src="https://github.com/user-attachments/assets/785a44e5-0254-480e-9b61-f1ae35ad9f33" />

12). Anomalous order Detection
``` sql

WITH invoice_totals AS (
    SELECT
    Invoice,
    CAST(customer_id AS INT64)      AS customer_id,
    DATE(InvoiceDate)               AS Invoice_Date,
    ROUND(SUM(Quantity * Price), 2) AS Invoice_Total
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY Invoice, customer_id, DATE(InvoiceDate)
),

customer_stats AS (
    SELECT
        customer_id,
        AVG(Invoice_Total)    AS Avg_Invoice,
        STDDEV(Invoice_Total) AS Std_Invoice,
        COUNT(*)              AS Total_Invoices
    FROM invoice_totals
    GROUP BY customer_id
    HAVING COUNT(*) >= 3                         
)

SELECT
    i.Invoice,
    i.customer_id,
    i.Invoice_Date,
    i.Invoice_Total,
    ROUND(cs.Avg_Invoice, 2)                   AS Customer_Avg_Invoice,
    ROUND(cs.Std_Invoice, 2)                   AS Customer_Std_Invoice,
    ROUND((i.Invoice_Total - cs.Avg_Invoice)/ NULLIF(cs.Std_Invoice, 0), 2) AS Z_Score,
    CASE
        WHEN (i.Invoice_Total - cs.Avg_Invoice)/ NULLIF(cs.Std_Invoice, 0) > 3    
        THEN 'ANOMALOUS_HIGH'
        WHEN (i.Invoice_Total - cs.Avg_Invoice)/ NULLIF(cs.Std_Invoice, 0) < -3   
        THEN 'ANOMALOUS_LOW'
        ELSE 'NORMAL'
    END AS Order_Classification
FROM invoice_totals i
JOIN customer_stats cs USING (customer_id)
WHERE ABS((i.Invoice_Total - cs.Avg_Invoice)
          / NULLIF(cs.Std_Invoice, 0)) > 3
ORDER BY Z_Score DESC;
```
<img width="1412" height="473" alt="image" src="https://github.com/user-attachments/assets/39095cc3-84ed-4faf-b640-6de96b7a941e" />

13). Day - hour Revenue Heatmap
``` sql
WITH HourlyMetrics AS (
    SELECT
        FORMAT_DATE('%A', DATE(InvoiceDate)) AS Day_Of_Week,
        EXTRACT(HOUR FROM InvoiceDate)       AS Hour_Of_Day,
        COUNT(DISTINCT Invoice)              AS Num_Transactions,
        ROUND(SUM(Quantity * Price), 2)      AS Total_Revenue,
        ROUND(AVG(Quantity * Price), 2)      AS Avg_Revenue_Per_Line,
        ROUND(SUM(Quantity * Price) / NULLIF(COUNT(DISTINCT Invoice), 0), 2) AS Avg_Revenue_Per_Invoice
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL 
      AND NOT STARTS_WITH(Invoice, 'C') 
      AND Quantity > 0 AND Price > 0
    GROUP BY 1, 2
)
SELECT 
    *,
    RANK() OVER (
        PARTITION BY Day_Of_Week 
        ORDER BY Avg_Revenue_Per_Line DESC
    ) AS Hour_Rank_Within_Day
FROM HourlyMetrics
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
```
<img width="1248" height="485" alt="image" src="https://github.com/user-attachments/assets/22c3eeb0-8439-4877-b54e-6293944156b3" />

14). Return MAtching and Net Revenue per Customer
``` sql
WITH purchases AS (
    SELECT
        CAST(customer_id AS INT64) AS customer_id,
        StockCode,
        SUM(Quantity) AS Units_Purchased,
        SUM(Quantity * Price) AS Gross_Revenue
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND customer_id IS NOT NULL
    GROUP BY 1, 2
),

returns AS (
    SELECT
        CAST(customer_id AS INT64) AS customer_id,
        StockCode,
        SUM(ABS(Quantity)) AS Units_Returned,
        SUM(ABS(Quantity * Price)) AS Return_Value
    FROM `customersegment.retail.segment`
    WHERE STARTS_WITH(Invoice, 'C')
      AND customer_id IS NOT NULL
    GROUP BY 1, 2
),

customer_net_revenue AS (
    SELECT
        p.customer_id,
        SUM(p.Gross_Revenue) AS Gross_Revenue,
        COALESCE(SUM(r.Return_Value), 0) AS Total_Returns,
        SUM(p.Gross_Revenue) - COALESCE(SUM(r.Return_Value), 0) AS Net_Revenue,
        SAFE_DIVIDE(COALESCE(SUM(r.Units_Returned), 0), SUM(p.Units_Purchased)) AS Return_Rate
    FROM purchases p
    LEFT JOIN returns r
        ON p.customer_id = r.customer_id
        AND p.StockCode = r.StockCode
    GROUP BY p.customer_id
)

SELECT
    customer_id,
    ROUND(Gross_Revenue, 2) AS Gross_Revenue,
    ROUND(Total_Returns, 2) AS Total_Returns,
    ROUND(Net_Revenue, 2) AS Net_Revenue,
    ROUND(Return_Rate * 100, 2) AS Return_Rate_Pct,
    CASE
        WHEN Net_Revenue < 0 THEN 'UNPROFITABLE'
        WHEN Return_Rate > 0.3 THEN 'HIGH_RETURN_RISK'
        WHEN Return_Rate > 0.1 THEN 'MODERATE_RETURN'
        ELSE 'LOW_RETURN'
    END AS Return_Risk_Flag
FROM customer_net_revenue
ORDER BY Return_Rate DESC;
```

<img width="1002" height="460" alt="image" src="https://github.com/user-attachments/assets/e9da9eeb-284e-4333-b0d9-cefccf231cf4" />

15). Return Rate by Product
``` sql

WITH product_sold AS (
    SELECT
        StockCode,
        TRIM(Description)                        AS Product,
        SUM(Quantity)                            AS Units_Sold,
        SUM(Quantity * Price)                    AS Revenue
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0
    GROUP BY StockCode, TRIM(Description)
),

product_returned AS (
    SELECT
        StockCode,
        SUM(ABS(Quantity))                       AS Units_Returned,
        SUM(ABS(Quantity * Price))               AS Return_Value
    FROM `customersegment.retail.segment`
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
```
<img width="1519" height="470" alt="image" src="https://github.com/user-attachments/assets/52b95b89-567b-488d-820c-e6888c7c3af1" /><img width="245" height="368" alt="image" src="https://github.com/user-attachments/assets/f9337d34-69f0-4106-8731-01c19cf45869" />

16). Bulk Buyer Identification

``` sql

WITH global_stats AS (
    SELECT
        AVG(Quantity)                            AS Global_Avg_Qty,
        STDDEV(Quantity)                         AS Global_Std_Qty
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND customer_id IS NOT NULL
),

customer_avg_qty AS (
    SELECT
        CAST(customer_id AS INT64)                AS customer_id,
        ROUND(AVG(Quantity), 2)                  AS Avg_Qty_Per_Line,
        ROUND(SUM(Quantity * Price), 2)          AS Total_Revenue,
        COUNT(DISTINCT Invoice)                  AS Total_Invoices,
        COUNT(*)                                 AS Total_Line_Items
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
      AND customer_id IS NOT NULL
    GROUP BY customer_id
)

SELECT
    c.customer_id,
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
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice,'C') AND Quantity > 0 AND Price > 0
),
buyer_classification AS (
    SELECT
        CAST(customer_id AS INT64) AS customer_id,
        AVG(Quantity) AS Avg_Qty,
        SUM(Quantity*Price) AS Revenue,
        COUNT(DISTINCT Invoice) AS Frequency
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice,'C') AND Quantity>0 AND Price>0
      AND customer_id IS NOT NULL
    GROUP BY customer_id
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
```
<img width="1368" height="485" alt="image" src="https://github.com/user-attachments/assets/e3f81608-7235-47b9-936d-32bbd442a22a" />

17). Month over Month Revenue growth by Country

``` sql
WITH raw_monthly_data AS (
    SELECT
        Country,
        DATE_TRUNC(DATE(InvoiceDate), MONTH)     AS Revenue_Month,
        SUM(Quantity * Price)                    AS Monthly_Revenue
    FROM `customersegment.retail.segment`
    WHERE NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY 1, 2
),

with_mom_calc AS (
    SELECT
        Country,
        Revenue_Month,
        Monthly_Revenue,
        LAG(Monthly_Revenue) OVER (
            PARTITION BY Country
            ORDER BY Revenue_Month
        )                                        AS Prev_Month_Revenue
    FROM raw_monthly_data
)

SELECT
    Country,
    Revenue_Month,
    ROUND(Monthly_Revenue, 2)                    AS Monthly_Revenue,
    ROUND(Prev_Month_Revenue, 2)                 AS Prev_Month_Revenue,
    -- Growth Calculation
    ROUND(
        SAFE_DIVIDE(Monthly_Revenue - Prev_Month_Revenue, Prev_Month_Revenue) * 100, 
        2
    )                                            AS MoM_Growth_Pct,
    -- Revenue Flagging Logic
    CASE
        WHEN SAFE_DIVIDE(Monthly_Revenue - Prev_Month_Revenue, Prev_Month_Revenue) < -0.20 
            THEN 'CRITICAL_DECLINE >20%'
        WHEN Monthly_Revenue < Prev_Month_Revenue 
            THEN 'DECLINING'
        WHEN SAFE_DIVIDE(Monthly_Revenue - Prev_Month_Revenue, Prev_Month_Revenue) > 0.20 
            THEN 'STRONG_GROWTH >20%'
        ELSE 'STABLE'
    END                                          AS Revenue_Flag
FROM with_mom_calc
WHERE Prev_Month_Revenue IS NOT NULL
ORDER BY Revenue_Month, MoM_Growth_Pct ASC;
```
<img width="1184" height="473" alt="image" src="https://github.com/user-attachments/assets/1391ab5a-f1fb-48d5-a5ec-686f73f296ad" />

18). Segemnet Revenue Contribution Summary

``` sql

WITH rfm_calculated AS (
    SELECT
        CAST(customer_id AS INT64)               AS customer_id,
        DATE_DIFF(DATE '2011-12-10',
                  DATE(MAX(InvoiceDate)), DAY)  AS Recency,
        COUNT(DISTINCT Invoice)                 AS Frequency,
        ROUND(SUM(Quantity * Price), 2)         AS Monetary
    FROM `customersegment.retail.segment`
    WHERE customer_id IS NOT NULL
      AND NOT STARTS_WITH(Invoice, 'C')
      AND Quantity > 0 AND Price > 0
    GROUP BY customer_id
),
scored AS (
    SELECT
        customer_id, Recency, Frequency, Monetary,
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
```
<img width="1254" height="353" alt="image" src="https://github.com/user-attachments/assets/979f7fcc-4515-489c-88d9-19b0613c5efa" /><img width="409" height="301" alt="image" src="https://github.com/user-attachments/assets/6235e861-cdd8-4aef-a21e-ed2c98fa793d" />


### Insights

- Pareto Efficiency: 20% of customers generate 77.26% of revenue. If the 'Champions' segment (roughly top 5-10%) is lost, the business would lose over 50% of its stable income.

- Return Impact: High-frequency buyers also tend to have higher return rates. However, the net monetary value of these customers remains positive, suggesting returns are a "cost of doing business" for loyalists.

- Anomaly Detection: Negative price entries (up to -£53,594) were identified as non-transactional adjustments and must be excluded from CLV models to avoid underestimation.

### Recommendations

- VIP Loyalty Program: Launch a dedicated referral program for 'Champions' to leverage their high brand affinity.

- Win-Back Automation: For 'At Risk' customers, trigger a personalized email at the 90-day inactivity mark, as the survival curve indicates a 50% churn probability after this point.

- Inventory for Q4: Given the strong Q4 seasonal preference found in 'Loyal' segments, increase stock depth for top-diversity products in September.

- Guest Conversion: Since Guest transactions (Missing IDs) have lower AOV, offer a first-purchase discount for account creation to track these currently "invisible" customers.

