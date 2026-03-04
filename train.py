import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore")

# Loading Data
print("Loading data...")
df = pd.read_csv("online_retail_II.csv", encoding="latin-1")
print(f"Dataset shape: {df.shape}")

#Cleaning Data
print("\nCleaning data...")

# Drop missing customer IDs
df = df.dropna(subset=["Customer ID"])
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# Add total price column
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Cleaned shape: {df.shape}")

# RFM Analysis 
print("\nCalculating RFM metrics...")

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,    # Recency
    "Invoice":     "nunique",                                   # Frequency
    "TotalPrice":  "sum"                                        # Monetary
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
print(f"RFM shape: {rfm.shape}")
print(rfm.describe())

# Scale Features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

#Train KMeans
print("\nTraining KMeans...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

#Label Clusters
cluster_summary = rfm.groupby("Cluster").agg({
    "Recency":   "mean",
    "Frequency": "mean",
    "Monetary":  "mean"
}).round(2)

print("\nCluster Summary:")
print(cluster_summary)

# Label clusters based on monetary value
cluster_order = cluster_summary["Monetary"].rank().astype(int)
labels = {0: "", 1: "", 2: "", 3: ""}
sorted_clusters = cluster_summary["Monetary"].sort_values()

tier_names = ["At-Risk Customers", "New Customers", "Loyal Customers", "Champion Customers"]
for i, (cluster_id, _) in enumerate(sorted_clusters.items()):
    labels[cluster_id] = tier_names[i]

rfm["Segment"] = rfm["Cluster"].map(labels)

print("\nSegment Distribution:")
print(rfm["Segment"].value_counts())

#Save Everything 
print("\nSaving artifacts...")
joblib.dump(kmeans,  "kmeans_model.pkl")
joblib.dump(scaler,  "rfm_scaler.pkl")
rfm.to_csv("rfm_data.csv", index=False)

print("\n Done! Files saved:")
print("  - kmeans_model.pkl")
print("  - rfm_scaler.pkl")
print("  - rfm_data.csv")