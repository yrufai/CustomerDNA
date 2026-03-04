import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Page config 
st.set_page_config(
    page_title="CustomerDNA",
    page_icon="🧬",
    layout="wide"
)
 
st.title("🧬 CustomerDNA")
st.markdown("Decode your customer base — identify Champions, Loyalists, and At-Risk customers instantly.")
st.divider()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("rfm_data.csv")

rfm = load_data()

# Segment colors
colors = {
    "Champion Customers": "#2ecc71",
    "Loyal Customers":    "#3498db",
    "New Customers":      "#f39c12",
    "At-Risk Customers":  "#e74c3c"
}

# Overview metrics 
st.subheader("Customer Overview")
col1, col2, col3, col4 = st.columns(4)

for col, segment, icon in zip(
    [col1, col2, col3, col4],
    ["Champion Customers", "Loyal Customers", "New Customers", "At-Risk Customers"],
    ["🏆", "💙", "🌱", "⚠️"]
):
    count = len(rfm[rfm["Segment"] == segment])
    pct   = count / len(rfm) * 100
    col.metric(f"{icon} {segment}", f"{count:,}", f"{pct:.1f}% of customers")

st.divider()

#Segment distribution chart
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Segment Distribution")
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_pie = px.pie(
        seg_counts, values="Count", names="Segment",
        color="Segment", color_discrete_map=colors,
        hole=0.4
    )
    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Revenue by Segment")
    revenue = rfm.groupby("Segment")["Monetary"].sum().reset_index()
    revenue.columns = ["Segment", "Total Revenue"]
    fig_bar = px.bar(
        revenue, x="Segment", y="Total Revenue",
        color="Segment", color_discrete_map=colors
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=0, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

#RFM Scatter 
st.subheader("RFM Analysis — Recency vs Monetary Value")
fig_scatter = px.scatter(
    rfm, x="Recency", y="Monetary",
    color="Segment", color_discrete_map=colors,
    size="Frequency", hover_data=["CustomerID", "Frequency"],
    opacity=0.7
)
fig_scatter.update_layout(margin=dict(t=0, b=0))
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Segment deep dive
st.subheader("Segment Deep Dive")
selected = st.selectbox("Select a segment to explore:", rfm["Segment"].unique())

filtered = rfm[rfm["Segment"] == selected]

c1, c2, c3 = st.columns(3)
c1.metric("Avg Recency (days)",  f"{filtered['Recency'].mean():.0f}")
c2.metric("Avg Frequency",       f"{filtered['Frequency'].mean():.1f}")
c3.metric("Avg Monetary Value",  f"${filtered['Monetary'].mean():,.2f}")

st.markdown(f"**{len(filtered):,} customers** in this segment")
st.dataframe(
    filtered[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]]
    .sort_values("Monetary", ascending=False)
    .head(20)
    .reset_index(drop=True),
    use_container_width=True
)

# Predict segment for new customer
st.divider()
st.subheader("Predict Segment for New Customer")

model  = joblib.load("kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")

col_a, col_b, col_c = st.columns(3)
recency   = col_a.number_input("Recency (days since last purchase)", 1, 800, 30)
frequency = col_b.number_input("Frequency (number of orders)", 1, 400, 5)
monetary  = col_c.number_input("Monetary (total spend $)", 1.0, 100000.0, 1000.0)

segment_labels = {
    cluster: rfm[rfm["Cluster"] == cluster]["Segment"].iloc[0]
    for cluster in rfm["Cluster"].unique()
}

if st.button(" Analyze Customer", use_container_width=True):
    input_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster      = model.predict(input_scaled)[0]
    segment      = segment_labels[cluster]
    color        = colors[segment]

    st.divider()
    if segment == "Champion Customers":
        st.success(f"🏆 **{segment}** — This is one of your best customers. Reward and retain them.")
    elif segment == "Loyal Customers":
        st.info(f"💙 **{segment}** — Reliable and engaged. Upsell premium products.")
    elif segment == "New Customers":
        st.warning(f"🌱 **{segment}** — Recently acquired. Nurture with onboarding offers.")
    else:
        st.error(f"⚠️ **{segment}** — At risk of churning. Re-engage with special offers immediately.")