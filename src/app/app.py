import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Customer Churn Prediction"
MODEL_PATH = "models/model.joblib"
PREPROCESSOR_PATH = "data/processed/preprocessor.joblib"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_churn(model, preprocessor, input_df: pd.DataFrame):
    X = preprocessor.transform(input_df)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1][0]
    else:
        # fallback
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))

    pred = int(prob >= 0.5)
    return prob, pred


def build_input_form():
    st.subheader("Customer details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])

    with col2:
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

    st.subheader("Billing")
    col3, col4 = st.columns(2)

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col4:
        monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
        total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=50000.0, value=2000.0, step=10.0)

    # Convert SeniorCitizen to expected format (dataset often uses 0/1)
    senior_val = 1 if senior == "Yes" else 0

    # Build one-row dataframe with the SAME column names as the dataset (minus Churn & customerID)
    row = {
        "gender": gender,
        "SeniorCitizen": senior_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📉", layout="centered")
    st.title("📉 Customer Churn Prediction")
    st.write(
        "Enter customer information to predict the probability of churn. "
        "This app uses a trained machine learning model and a preprocessing pipeline."
    )

    with st.expander("✅ What this project demonstrates", expanded=False):
        st.markdown(
            "- Reproducible data pipeline (download + preprocess)\n"
            "- Training + evaluation with saved artifacts\n"
            "- Production-style preprocessing with sklearn\n"
            "- Streamlit app for interactive inference"
        )

    model, preprocessor = load_artifacts()

    input_df = build_input_form()

    st.divider()
    if st.button("Predict churn", type="primary"):
        prob, pred = predict_churn(model, preprocessor, input_df)

        st.subheader("Prediction result")
        st.metric("Churn probability", f"{prob:.2%}")

        st.progress(min(max(prob, 0.0), 1.0))

        if pred == 1:
            st.error("⚠️ High risk: Customer is likely to churn.")
            st.markdown(
                "**Suggested actions:**\n"
                "- Offer retention discount\n"
                "- Improve support / tech assistance\n"
                "- Propose annual contract upgrade"
            )
        else:
            st.success("✅ Low risk: Customer is not likely to churn.")
            st.markdown(
                "**Suggested actions:**\n"
                "- Encourage referrals\n"
                "- Offer add-on services\n"
                "- Maintain good support quality"
            )

        with st.expander("See input data"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()
