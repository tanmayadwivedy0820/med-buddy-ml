import requests
import pandas as pd
import streamlit as st

import config

st.title("🧠 Model Insights")
st.write("How the trained model makes decisions — its configuration and which features drive predictions.")

st.divider()


@st.cache_data(ttl=300)
def fetch_model_info():
    response = requests.get(config.MODEL_INFO_URL, timeout=10)
    response.raise_for_status()
    return response.json()


try:
    info = fetch_model_info()
except requests.exceptions.RequestException:
    st.error("Could not load model info. Is the backend running?")
    st.stop()

# --- Model configuration ---
st.subheader("Model Configuration")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", info["model_type"])
c2.metric("Trees", f"{info['n_estimators']:,}")
c3.metric("Max Depth", info["max_depth"])
c4.metric("Features", info["n_features"])

st.divider()

# --- Feature importances ---
st.subheader("Feature Importances")
st.caption(
    "Impurity-based (Gini) importance from the Random Forest. "
    "Higher means the feature contributed more to the model's splits."
)

fi = info["feature_importances"]  # already ranked descending by the API
fi_df = pd.DataFrame(fi).set_index("feature")

# Horizontal bar chart, most important at top
st.bar_chart(fi_df, horizontal=True)

st.divider()

# --- Ranked table ---
st.subheader("Ranked Importance Table")
table_df = fi_df.copy()
table_df["importance"] = (table_df["importance"] * 100).round(2).astype(str) + " %"
st.dataframe(table_df, width="stretch")

st.info(
    "Note: impurity-based importances can be biased toward high-cardinality features. "
    "Permutation importance or SHAP values would give a more robust attribution.",
    icon="ℹ️",
)