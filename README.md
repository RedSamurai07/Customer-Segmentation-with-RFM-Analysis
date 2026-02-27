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

Python: Used for data cleaning, advanced feature engineering, and machine learning. Libraries: Pandas, Numpy, Scikit-learn (K-Means, GMM, Agglomerative), Scipy (Stats), Matplotlib, Seaborn.

SQL: Used for production-ready queries including Cohort Analysis, Window Functions for Pareto thresholds, and Rolling Retention.

Excel/CSV: Initial data inspection and output storage.

Tableau: Visualization for the final retail data.
  
### Analysis
**Python**


**Hypothesis Testing**


Champions vs. At-Risk (Monetary Value):Test: Mann-Whitney U Test (due to non-normal distribution).Result: $p < 0.001$, Cohen’s $d = 0.563$.Insight: There is a statistically significant difference in spend between these segments, confirming that our segmentation logic effectively separates high-value and low-value groups.Country Distribution:Test: Kruskal-Wallis Test.Result: Significant differences found ($p < 0.05$). The UK dominates in volume, but countries like EIRE and Netherlands show higher average spend per customer.

**SQL**


**Tableau**



### Insights

- Pareto Efficiency: 20% of customers generate 77.26% of revenue. If the 'Champions' segment (roughly top 5-10%) is lost, the business would lose over 50% of its stable income.

- Return Impact: High-frequency buyers also tend to have higher return rates. However, the net monetary value of these customers remains positive, suggesting returns are a "cost of doing business" for loyalists.

- Anomaly Detection: Negative price entries (up to -£53,594) were identified as non-transactional adjustments and must be excluded from CLV models to avoid underestimation.

### Recommendations

- VIP Loyalty Program: Launch a dedicated referral program for 'Champions' to leverage their high brand affinity.

- Win-Back Automation: For 'At Risk' customers, trigger a personalized email at the 90-day inactivity mark, as the survival curve indicates a 50% churn probability after this point.

- Inventory for Q4: Given the strong Q4 seasonal preference found in 'Loyal' segments, increase stock depth for top-diversity products in September.

- Guest Conversion: Since Guest transactions (Missing IDs) have lower AOV, offer a first-purchase discount for account creation to track these currently "invisible" customers.

