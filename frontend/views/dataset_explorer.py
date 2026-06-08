import requests
import pandas as pd
import streamlit as st

import config

st.title("📊 Dataset Explorer")
st.write("Exploratory analysis of the heart disease dataset the model was trained on.")

st.divider()


# Cache the API call so switching pages / re-running doesn't re-fetch every time.
# ttl=300 means the cached result is reused for 5 minutes.
@st.cache_data(ttl=300)
def fetch_dataset_stats():
    response = requests.get(config.DATASET_STATS_URL, timeout=10)
    response.raise_for_status()
    return response.json()


try:
    stats = fetch_dataset_stats()
except requests.exceptions.RequestException:
    st.error("Could not load dataset stats. Is the backend running?")
    st.stop()

# --- Top-level metrics ---
m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{stats['n_rows']:,}")
m2.metric("Features", stats["n_features"])
balance = stats["target_balance"]
m3.metric("Positive Cases", f"{balance.get('1', 0):,}")

st.divider()

# --- Target balance ---
st.subheader("Target Class Balance")
st.caption("How many patients in the dataset have heart disease (1) vs not (0).")
balance_df = pd.DataFrame(
    {"count": [balance.get("0", 0), balance.get("1", 0)]},
    index=["No Disease (0)", "Disease (1)"],
)
st.bar_chart(balance_df)

st.divider()

# --- Feature distribution explorer ---
st.subheader("Feature Distributions")
st.caption("Select a feature to see how its values are distributed across the dataset.")

feature = st.selectbox("Feature", stats["feature_names"])

hist = stats["histograms"][feature]
# bin_edges has N+1 entries for N bins; build a readable label per bin
edges = hist["bin_edges"]
labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(hist["counts"]))]
hist_df = pd.DataFrame({"count": hist["counts"]}, index=labels)
st.bar_chart(hist_df)

st.divider()

# --- Summary statistics table ---
st.subheader("Summary Statistics")
st.caption("Standard descriptive statistics per feature (from pandas .describe()).")
summary_df = pd.DataFrame(stats["summary"]).T  # transpose: features as rows
st.dataframe(summary_df, width="stretch")