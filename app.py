import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
art = joblib.load("models/churn.pkl"); model, th = art["model"], art["threshold"]
st.title("📉 Customer Churn Predictor")

yn, yni = ["No","Yes"], ["No","Yes","No internet service"]
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female","Male"])
    senior = st.selectbox("Senior citizen", [0,1])
    partner = st.selectbox("Partner", yn); dep = st.selectbox("Dependents", yn)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", yn)
    lines = st.selectbox("Multiple lines", ["No","Yes","No phone service"])
    net = st.selectbox("Internet service", ["DSL","Fiber optic","No"])
    sec = st.selectbox("Online security", yni)
with c2:
    bak = st.selectbox("Online backup", yni); dev = st.selectbox("Device protection", yni)
    tech = st.selectbox("Tech support", yni); tv = st.selectbox("Streaming TV", yni)
    mov = st.selectbox("Streaming movies", yni)
    con = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    pap = st.selectbox("Paperless billing", yn)
    pay = st.selectbox("Payment method", ["Electronic check","Mailed check",
                       "Bank transfer (automatic)","Credit card (automatic)"])
    mc = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
tc = tenure * mc

if st.button("Predict"):
    row = pd.DataFrame([dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dep,
        tenure=tenure, PhoneService=phone, MultipleLines=lines, InternetService=net,
        OnlineSecurity=sec, OnlineBackup=bak, DeviceProtection=dev, TechSupport=tech,
        StreamingTV=tv, StreamingMovies=mov, Contract=con, PaperlessBilling=pap,
        PaymentMethod=pay, MonthlyCharges=mc, TotalCharges=tc)])
    p = model.predict_proba(row)[0,1]
    st.metric("Churn probability", f"{p:.1%}")
    (st.error if p >= th else st.success)("⚠️ High churn risk" if p >= th else "✅ Likely to stay")