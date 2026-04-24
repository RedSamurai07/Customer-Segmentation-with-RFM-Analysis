import streamlit as st
import pandas as pd
import os
from train import preprocess_data, calculate_rfm, segment_customers

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation with RFM Analysis")

st.write("""
Upload your transaction dataset to perform RFM (Recency, Frequency, Monetary) analysis and segment your customers into Top, Middle, and Low tiers.
""")

uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        if st.button("Run Segmentation"):
            with st.spinner("Processing data and calculating RFM..."):
                # Run the functions from train.py
                df_clean = preprocess_data(df)
                rfm = calculate_rfm(df_clean)
                rfm_final = segment_customers(rfm)
                
                st.success("Segmentation complete!")
                
                st.write("### Segmentation Results")
                st.dataframe(rfm_final)
                
                st.write("### Segment Summary")
                summary = rfm_final.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'MonetaryValue': ['mean', 'count']
                }).round(1)
                st.dataframe(summary)
                
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload a dataset to begin. The dataset should contain columns like 'Customer ID', 'Quantity', 'Price', 'Invoice', and 'InvoiceDate'.")
