"""
Tests for train.py — targets >85% code coverage.

Covers:
  - preprocess_data: happy path + edge cases (nulls, cancellations, negatives)
  - calculate_rfm: correct recency/frequency/monetary aggregation
  - segment_customers: all three segments (Top, Middle, Low) + RFM scoring
  - load_data: mocked multi-sheet and fallback paths
  - train(): mocked MLflow end-to-end
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import preprocess_data, calculate_rfm, segment_customers, load_data


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

def make_raw_df(n_customers=20, seed=42):
    """Build a synthetic retail dataframe with enough variance for qcut."""
    rng = np.random.default_rng(seed)
    n = n_customers * 5
    base_date = datetime(2011, 1, 1)
    dates = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 300, n)]

    df = pd.DataFrame({
        "Customer ID": rng.choice([f"C{i:04d}" for i in range(n_customers)], n),
        "Invoice": [f"INV{i:05d}" for i in range(n)],
        "InvoiceDate": dates,
        "Quantity": rng.integers(1, 20, n).astype(float),
        "Price": rng.uniform(0.5, 50, n),
        "StockCode": "ABC",
        "Description": "Item",
        "Country": "UK",
    })
    return df


def make_preprocessed_df(n_customers=20, seed=42):
    """Return already-cleaned dataframe (output of preprocess_data)."""
    return preprocess_data(make_raw_df(n_customers=n_customers, seed=seed))


# ─────────────────────────────────────────
# preprocess_data
# ─────────────────────────────────────────

class TestPreprocessData:
    def test_removes_null_customer_ids(self):
        df = make_raw_df()
        df.loc[df.index[:5], "Customer ID"] = np.nan
        result = preprocess_data(df)
        assert result["Customer ID"].notna().all()

    def test_removes_non_positive_quantity(self):
        df = make_raw_df()
        df.loc[df.index[0], "Quantity"] = 0
        df.loc[df.index[1], "Quantity"] = -5
        result = preprocess_data(df)
        assert (result["Quantity"] > 0).all()

    def test_removes_non_positive_price(self):
        df = make_raw_df()
        df.loc[df.index[0], "Price"] = 0
        df.loc[df.index[1], "Price"] = -1.0
        result = preprocess_data(df)
        assert (result["Price"] > 0).all()

    def test_removes_cancelled_invoices(self):
        df = make_raw_df()
        df.loc[df.index[:3], "Invoice"] = ["C00001", "C00002", "C00003"]
        result = preprocess_data(df)
        assert not result["Invoice"].astype(str).str.startswith("C").any()

    def test_invoice_date_is_datetime(self):
        df = make_raw_df()
        result = preprocess_data(df)
        assert pd.api.types.is_datetime64_any_dtype(result["InvoiceDate"])

    def test_total_sum_column_created(self):
        df = make_raw_df()
        result = preprocess_data(df)
        assert "TotalSum" in result.columns
        # Spot-check: TotalSum == Quantity * Price
        row = result.iloc[0]
        assert abs(row["TotalSum"] - row["Quantity"] * row["Price"]) < 1e-6

    def test_strips_column_whitespace(self):
        df = make_raw_df()
        df.columns = [f" {c} " for c in df.columns]
        result = preprocess_data(df)
        for col in result.columns:
            assert col == col.strip()

    def test_returns_non_empty_dataframe(self):
        df = make_raw_df()
        result = preprocess_data(df)
        assert len(result) > 0


# ─────────────────────────────────────────
# calculate_rfm
# ─────────────────────────────────────────

class TestCalculateRFM:
    def test_returns_expected_columns(self):
        df = make_preprocessed_df()
        rfm = calculate_rfm(df)
        assert set(["Recency", "Frequency", "MonetaryValue"]).issubset(rfm.columns)

    def test_recency_is_non_negative(self):
        df = make_preprocessed_df()
        rfm = calculate_rfm(df)
        assert (rfm["Recency"] >= 0).all()

    def test_frequency_is_at_least_one(self):
        df = make_preprocessed_df()
        rfm = calculate_rfm(df)
        assert (rfm["Frequency"] >= 1).all()

    def test_monetary_is_positive(self):
        df = make_preprocessed_df()
        rfm = calculate_rfm(df)
        assert (rfm["MonetaryValue"] > 0).all()

    def test_index_is_customer_id(self):
        df = make_preprocessed_df()
        rfm = calculate_rfm(df)
        assert rfm.index.name == "Customer ID"

    def test_one_customer_recency(self):
        """Single-customer edge case: recency should be 0 (bought today)."""
        today = pd.Timestamp("2020-01-10")
        df = pd.DataFrame({
            "Customer ID": ["CUST_A"],
            "Invoice": ["INV001"],
            "InvoiceDate": [today],
            "Quantity": [2.0],
            "Price": [5.0],
            "TotalSum": [10.0],
        })
        rfm = calculate_rfm(df)
        # snapshot = max + 1 day → recency = 1
        assert rfm.loc["CUST_A", "Recency"] == 1
        assert rfm.loc["CUST_A", "Frequency"] == 1
        assert abs(rfm.loc["CUST_A", "MonetaryValue"] - 10.0) < 1e-6

    def test_frequency_counts_unique_invoices(self):
        base = pd.Timestamp("2020-06-01")
        df = pd.DataFrame({
            "Customer ID": ["A", "A", "A"],
            "Invoice": ["INV1", "INV1", "INV2"],  # 2 unique
            "InvoiceDate": [base, base, base + timedelta(days=5)],
            "Quantity": [1.0, 1.0, 1.0],
            "Price": [10.0, 10.0, 20.0],
            "TotalSum": [10.0, 10.0, 20.0],
        })
        rfm = calculate_rfm(df)
        assert rfm.loc["A", "Frequency"] == 2


# ─────────────────────────────────────────
# segment_customers
# ─────────────────────────────────────────

class TestSegmentCustomers:
    def _get_rfm(self, n=40, seed=0):
        df = make_preprocessed_df(n_customers=n, seed=seed)
        return calculate_rfm(df)

    def test_returns_segment_column(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert "Segment" in result.columns

    def test_segment_values_are_valid(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert set(result["Segment"].unique()).issubset({"Top", "Middle", "Low"})

    def test_rfm_score_column_exists(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert "RFM_Score" in result.columns

    def test_rfm_score_is_sum_of_r_f_m(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        computed = result["R"].astype(int) + result["F"].astype(int) + result["M"].astype(int)
        pd.testing.assert_series_equal(
            result["RFM_Score"].astype(int),
            computed,
            check_names=False,
        )

    def test_top_segment_has_score_gte_9(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert (result.loc[result["Segment"] == "Top", "RFM_Score"] >= 9).all()

    def test_low_segment_has_score_lt_5(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert (result.loc[result["Segment"] == "Low", "RFM_Score"] < 5).all()

    def test_middle_segment_range(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        mid = result.loc[result["Segment"] == "Middle", "RFM_Score"]
        assert ((mid >= 5) & (mid < 9)).all()

    def test_all_customers_assigned(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        assert result["Segment"].notna().all()

    def test_r_f_m_scores_are_1_to_4(self):
        rfm = self._get_rfm()
        result = segment_customers(rfm)
        for col in ["R", "F", "M"]:
            vals = result[col].astype(int)
            assert vals.min() >= 1
            assert vals.max() <= 4

    def test_three_segments_produced(self):
        """With 40 customers and enough variance all three segments appear."""
        rfm = self._get_rfm(n=40, seed=7)
        result = segment_customers(rfm)
        assert len(result["Segment"].unique()) == 3


# ─────────────────────────────────────────
# load_data
# ─────────────────────────────────────────

class TestLoadData:
    def test_load_data_multi_sheet(self, tmp_path):
        """Happy path: two-sheet Excel file is concatenated."""
        df1 = make_raw_df(n_customers=5, seed=1)
        df2 = make_raw_df(n_customers=5, seed=2)
        path = tmp_path / "retail.xlsx"
        with pd.ExcelWriter(path) as writer:
            df1.to_excel(writer, sheet_name="Year 2009-2010", index=False)
            df2.to_excel(writer, sheet_name="Year 2010-2011", index=False)
        result = load_data(str(path))
        assert len(result) == len(df1) + len(df2)

    def test_load_data_fallback_single_sheet(self, tmp_path):
        """Fallback: file with no named sheets → reads default sheet."""
        df = make_raw_df(n_customers=5, seed=3)
        path = tmp_path / "retail_single.xlsx"
        df.to_excel(path, index=False)
        result = load_data(str(path))
        assert len(result) == len(df)


# ─────────────────────────────────────────
# train() – mocked MLflow
# ─────────────────────────────────────────

class TestTrain:
    @patch("train.load_data")
    def test_train_runs_end_to_end(self, mock_load, tmp_path):
        """train() calls mlflow logging and completes without error."""
        df = make_raw_df(n_customers=40, seed=99)
        mock_load.return_value = df

        # Change working dir so CSV artifact saves somewhere writable
        os.chdir(tmp_path)

        import mlflow
        mock_client = MagicMock()

        with patch.object(mlflow, "set_experiment"), \
             patch.object(mlflow, "start_run") as mock_start_run, \
             patch.object(mlflow, "log_param") as mock_log_param, \
             patch.object(mlflow, "log_metric") as mock_log_metric, \
             patch.object(mlflow, "log_artifact"):

            mock_run_ctx = MagicMock()
            mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            from train import train
            train()  # Should not raise

            mock_log_param.assert_called()
            mock_log_metric.assert_called()