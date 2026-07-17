import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_prediction_model.pkl"

model = joblib.load(MODEL_PATH)

## page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction & Retention Intelligence Platform")

st.write(
    "Upload a customer dataset to predict churn probability and identify high-risk customers."
)

uploaded_file = st.file_uploader(
    "Upload Customer Dataset (.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if st.button("Predict Churn"):

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        data["Prediction"] = predictions
        data["Churn Probability"] = probabilities

        def risk(prob):
            if prob >= 0.75:
                return "🔴 High"
            elif prob >= 0.40:
                return "🟡 Medium"
            else:
                return "🟢 Low"

        data["Risk Level"] = data["Churn Probability"].apply(risk)

        st.subheader("Prediction Results")
        st.dataframe(data)

        ## displaying KPIs
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(data))

        col2.metric(
            "High Risk Customers",
            (data["Risk Level"] == "🔴 High").sum()
        )

        col3.metric(
            "Average Churn Probability",
            f"{data['Churn Probability'].mean()*100:.2f}%"
        )

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Predictions",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

        ## churn probability chart
        fig = px.histogram(
            data,
            x="Churn Probability",
            nbins=20,
            title="Churn Probability Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        ## risk level pie chart
        fig = px.pie(
            data,
            names="Risk Level",
            title="Customer Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        ## search customer
        customer = st.text_input("Search Customer ID")

        if customer:
            st.dataframe(data[data["CustomerID"] == customer])

        ## high risk customers
        if st.checkbox("Show High Risk Customers Only"):
            st.dataframe(data[data["Risk Level"] == "🔴 High"])

st.markdown("---")
st.markdown("Developed by Sandeep Singh Chouhan")
