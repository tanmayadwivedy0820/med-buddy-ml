import streamlit as st

import config

# Global page config — set ONCE here, in the entry point.
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
)

# Declare pages explicitly using the modern st.navigation API.
# Each st.Page points to a file in views/ — Streamlit runs that
# file's top-level code when the page is selected.
predictor_page = st.Page(
    "views/predictor.py",
    title="Predictor",
    icon="🔍",
    default=True,
)
dataset_page = st.Page(
    "views/dataset_explorer.py",
    title="Dataset Explorer",
    icon="📊",
)
model_page = st.Page(
    "views/model_insights.py",
    title="Model Insights",
    icon="🧠",
)

# Build the navigation. Grouping gives a labeled sidebar section.
nav = st.navigation(
    {
        "MedBuddy.ML": [predictor_page, dataset_page, model_page],
    }
)

# Sidebar branding shown on every page
st.sidebar.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.sidebar.caption("Heart Disease ML Dashboard")

# Hand control to the selected page
nav.run()