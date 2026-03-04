# CustomerDNA 🧬

Customer segmentation using RFM analysis and K-Means clustering, with an interactive Streamlit dashboard.

---

## Overview

CustomerDNA takes raw e-commerce transaction data and groups customers into four behavioral segments — Champions, Loyal, New, and At-Risk — based on how recently they bought, how often, and how much they spent. The dashboard lets you explore each segment and predict where any new customer belongs.

---

## Segments

| | Segment | Profile |
|---|---|---|
| 🏆 | Champion Customers | High spenders who buy often and recently |
| 💙 | Loyal Customers | Consistent buyers, strong lifetime value |
| 🌱 | New Customers | Recently acquired, still building habit |
| ⚠️ | At-Risk Customers | Haven't bought in a while, low engagement |

---

## Setup

**Requirements:** Python 3.8+, and `online_retail_II.csv` in the project root.

```bash
pip install -r requirements.txt
```

Train the model (only needs to run once):

```bash
python train.py
```

This processes the raw data, runs RFM + K-Means, and saves three files: `rfm_data.csv`, `kmeans_model.pkl`, and `rfm_scaler.pkl`.

Launch the dashboard:

```bash
streamlit run app.py
```

---

## How the model works

`train.py` does the following:

1. Cleans the raw data — drops missing customer IDs, strips cancelled orders, removes bad quantities/prices
2. Computes RFM metrics per customer (Recency in days, Frequency as unique order count, Monetary as total spend)
3. Scales features with `StandardScaler`
4. Fits a K-Means model with 4 clusters
5. Labels clusters by ranking their average monetary value

---

## Dashboard

The app (`app.py`) has five views:

- **Overview** — headcount and percentage breakdown per segment
- **Distribution** — donut chart and a revenue-by-segment bar chart
- **RFM Scatter** — Recency vs. Monetary, bubble size = Frequency
- **Deep Dive** — drill into any segment, see average RFM stats and top customers
- **Predict** — enter Recency / Frequency / Monetary values to classify a new customer

---

## Stack

Streamlit · scikit-learn · pandas · Plotly · joblib
