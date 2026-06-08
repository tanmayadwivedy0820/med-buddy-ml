import requests
import streamlit as st

import config

st.title("🔍 Heart Disease Predictor")
st.write("Enter patient details and click **Predict** to estimate heart disease risk.")

st.divider()

# --- Input form ---
# Wrapping inputs in a form means the API is only called on submit,
# not on every single widget change (Streamlit reruns on each interaction).
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 52)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
        cp = st.number_input("Chest Pain Type (cp)", 0, 3, 0)
        trestbps = st.number_input("Resting Blood Pressure", 0, 250, 125)
        chol = st.number_input("Cholesterol", 0, 600, 212)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.number_input("Resting ECG (restecg)", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate (thalach)", 0, 250, 168)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])

    with col3:
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
        slope = st.number_input("Slope", 0, 2, 2)
        ca = st.number_input("Number of Major Vessels (ca)", 0, 4, 0)
        thal = st.number_input("Thal", 0, 3, 2)

    submitted = st.form_submit_button("🔍 Predict", width="stretch")

# --- Handle submission ---
if submitted:
    input_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    try:
        with st.spinner("Contacting model..."):
            response = requests.post(config.PREDICT_URL, json=input_data, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Is the backend running on the configured host?")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The request timed out. Try again.")
        st.stop()
    except requests.exceptions.HTTPError:
        st.error(f"API returned an error (status {response.status_code}).")
        st.stop()

    result = response.json()
    prediction = result["prediction"]
    probability = result["probability"]
    diagnosis = result["diagnosis"]

    st.divider()
    st.subheader("Result")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Heart Disease Probability", value=f"{probability:.1%}")
    with res_col2:
        st.progress(probability)

    if prediction == 1:
        st.error(f"⚠️ Model Prediction: {diagnosis}")
    else:
        st.success(f"✅ Model Prediction: {diagnosis}")

    st.caption(
        "This is a machine-learning estimate for demonstration only and is not medical advice."
    )