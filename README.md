# CustomerDNA 🧬
> Customer segmentation using RFM analysis and K-Means clustering, deployed live on AWS EC2.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-customerdna.yrufai.com-blue)](http://customerdna.yrufai.com)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io)
[![AWS](https://img.shields.io/badge/Deployed%20on-AWS%20EC2-orange)](https://aws.amazon.com)

---

## 🔗 Live Demo
**[customerdna.yrufai.com](http://customerdna.yrufai.com)**

---

## Overview
CustomerDNA analyzes over **1 million real UK e-commerce transactions** and groups **5,878 customers** into four behavioral segments — Champions, Loyal, New, and At-Risk — based on how recently they bought, how often, and how much they spent.

The interactive dashboard lets you explore each segment, analyze revenue distribution, and predict where any new customer belongs in real time.

---

## Segments

| | Segment | Profile |
|---|---|---|
| 🏆 | Champion Customers | High spenders who buy often and recently |
| 💙 | Loyal Customers | Consistent buyers, strong lifetime value |
| 🌱 | New Customers | Recently acquired, still building habit |
| ⚠️ | At-Risk Customers | Haven't bought in a while, low engagement |

---

## How It Works

CustomerDNA uses **RFM Analysis** — a proven marketing technique that scores customers on three dimensions:

- **Recency** — How recently did they purchase? (days since last order)
- **Frequency** — How often do they buy? (number of unique orders)
- **Monetary** — How much do they spend? (total revenue generated)

Once RFM scores are computed, **K-Means clustering** groups customers into 4 segments automatically — no manual rules needed.

### Model Pipeline (`train.py`)
1. Load and clean 1M+ raw transactions — drop nulls, cancelled orders, bad quantities
2. Compute RFM metrics per customer
3. Normalize features using `StandardScaler`
4. Fit K-Means with 4 clusters
5. Label segments by average monetary value
6. Save model artifacts: `kmeans_model.pkl`, `rfm_scaler.pkl`, `rfm_data.csv`

---

## Dashboard Features

The app (`app.py`) has five interactive views:

| View | Description |
|---|---|
| **Overview** | Headcount and percentage breakdown per segment |
| **Distribution** | Donut chart and revenue-by-segment bar chart |
| **RFM Scatter** | Recency vs. Monetary, bubble size = Frequency |
| **Deep Dive** | Drill into any segment, see top customers and avg RFM stats |
| **Predict** | Enter R/F/M values to classify any new customer in real time |

---

## Dataset

- **Source:** [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Size:** 1,067,371 transactions → cleaned to 805,549
- **Customers:** 5,878 unique customers analyzed
- **Period:** December 2009 – December 2011

> Note: The dataset file (`online_retail_II.csv`) is not included in this repo due to size. Download it from the link above and place it in the project root before running `train.py`. The pre-trained model files (`kmeans_model.pkl`, `rfm_scaler.pkl`, `rfm_data.csv`) are included so you can run the dashboard without retraining.

---

## Setup

**Requirements:** Python 3.8+

```bash
git clone https://github.com/yrufai/CustomerDNA.git
cd CustomerDNA
pip install -r requirements.txt
```

Launch the dashboard (model already trained):
```bash
streamlit run app.py
```

To retrain from scratch (requires dataset):
```bash
python train.py
streamlit run app.py
```

---

## Project Structure

```
CustomerDNA/
├── app.py               # Streamlit dashboard
├── train.py             # RFM analysis and K-Means training
├── requirements.txt     # Dependencies
├── rfm_data.csv         # Pre-computed RFM scores and segments
├── kmeans_model.pkl     # Trained K-Means model
└── rfm_scaler.pkl       # Fitted StandardScaler
```

---

## Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas + NumPy | Data processing and RFM computation |
| scikit-learn | K-Means clustering and StandardScaler |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| AWS EC2 | Cloud deployment |

---

## Author

**Rufai Yakubu** — [yrufai.com](http://yrufai.com) · [GitHub](https://github.com/yrufai) · [LinkedIn](https://linkedin.com/in/yrufai)