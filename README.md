# med-buddy-## Architecture

MedBuddy.ML is split into two decoupled layers communicating over HTTP:

**Backend (FastAPI)** — owns the model and data.
- `/predict-heart-disease` — runs inference on patient input
- `/dataset-stats` — read-only EDA data (summary stats, distributions, class balance)
- `/model-info` — model config and feature importances
- The trained model (a `StandardScaler → RandomForestClassifier` pipeline) is
  loaded once at startup as a module-level singleton and shared across endpoints.

**Frontend (Streamlit, multipage)** — pure presentation, no ML logic.
- Built with the `st.navigation` / `st.Page` API; `app.py` is a thin router.
- `config.py` holds a single `API_BASE_URL`; each page composes its own endpoint.
- Three pages: Predictor (form → API), Dataset Explorer (EDA dashboard),
  Model Insights (feature importances + model config).
- API responses are cached with `st.cache_data` to avoid redundant calls on rerun.

The frontend never touches the CSV or the model artifact directly — everything
goes through the API. Swapping the model (e.g. RandomForest → XGBoost) requires
no frontend changes.

## Running locally

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/app.py
```