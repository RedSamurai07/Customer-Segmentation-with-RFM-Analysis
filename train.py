import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime

def load_data(file_path):
    """Loads the retail dataset."""
    try:
        df_1 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
        df_2 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
        df = pd.concat([df_1, df_2], ignore_index=True)
    except Exception:
        df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    """Cleans the dataset."""
    df.columns = df.columns.str.strip()
    # Drop rows without Customer ID, non-positive quantity or price
    df = df[df['Customer ID'].notna()]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    # Filter for valid purchases (exclude cancellations)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['Price']
    return df

def calculate_rfm(df):
    """Calculates R, F, M metrics."""
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalSum': 'sum'
    })
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalSum': 'MonetaryValue'
    }, inplace=True)
    return rfm

def segment_customers(rfm):
    """Assigns scores and segments."""
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    r_groups = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    f_groups = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
    m_groups = pd.qcut(rfm['MonetaryValue'], q=4, labels=m_labels)
    
    rfm = rfm.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
    
    def get_segment(df):
        if df['RFM_Score'] >= 9:
            return 'Top'
        elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
            return 'Middle'
        else:
            return 'Low'
            
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    return rfm

def train():
    mlflow.set_experiment("Customer_Segmentation")
    
    with mlflow.start_run():
        print("Loading data...")
        df = load_data('online_retail_II.xlsx')
        
        print("Preprocessing data...")
        df_clean = preprocess_data(df)
        
        print("Calculating RFM...")
        rfm = calculate_rfm(df_clean)
        
        print("Segmenting customers...")
        rfm_final = segment_customers(rfm)
        
        # Log parameters
        mlflow.log_param("num_customers", len(rfm_final))
        mlflow.log_param("data_source", "online_retail_II.xlsx")
        
        # Log metrics
        avg_recency = rfm_final['Recency'].mean()
        avg_frequency = rfm_final['Frequency'].mean()
        avg_monetary = rfm_final['MonetaryValue'].mean()
        
        mlflow.log_metric("avg_recency", avg_recency)
        mlflow.log_metric("avg_frequency", avg_frequency)
        mlflow.log_metric("avg_monetary", avg_monetary)
        
        # Save and log results summary
        summary = rfm_final.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'MonetaryValue': ['mean', 'count']
        }).round(1)
        
        summary.to_csv("segmentation_summary.csv")
        mlflow.log_artifact("segmentation_summary.csv")
        
        print("Run complete. Summary logged to MLflow.")

if __name__ == "__main__":
    train()
