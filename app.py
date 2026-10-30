
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

# App heading
st.title("📊 Customer Churn Prediction")
st.write(
    "An ML-powered dashboard that estimates whether "
    "a telecom customer is likely to leave the service."
)

st.info(
    "This is an educational prediction tool, "
    "not a guarantee of actual customer behaviour."
)

# Input form
st.header("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

    with col2:
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0, max_value=200.0, value=70.0
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0, max_value=10000.0, value=840.0
        )

    submit = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

# Make a prediction when submitted
if submit:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = model.predict_proba(customer)[0][1]
    prediction = model.predict(customer)[0]

    st.divider()
    st.header("Prediction Result")

    left, right = st.columns(2)

    with left:
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )
        st.progress(float(probability))

    with right:
        if prediction == 1:
            st.error("Prediction: Customer may churn")
        else:
            st.success("Prediction: Customer may stay")

    st.caption(
        "Predictions are based on patterns in historical data. "
        "They should not be treated as certain outcomes."
    )
