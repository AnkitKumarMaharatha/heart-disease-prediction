import streamlit as st
import pandas as pd
import joblib

# 1. Load your artifacts safely
model = joblib.load("Log_Heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
columns = joblib.load("columns_heart.pkl")

# Configured to wide layout for an interactive dashboard experience
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Diagnostic Assistant")
st.write("An interactive clinical decision-support panel driven by predictive machine learning algorithms.")
st.markdown("---")

# Setup application routing using interactive tabs
tab1, tab2 = st.tabs(["📊 Diagnostic Dashboard", "ℹ️ Documentation & Guidelines"])

with tab2:
    st.markdown("""
    ### Interactive Tool Guidelines
    1. Fill in patient information inside the **Demographics & Symptoms** block.
    2. Provide physical measurements and clinical lab readings inside the **Physiological Metrics** block.
    3. Click **Run Diagnostic Prediction** to pass the data pipeline into the standardized machine learning engine.
    
    *Note: Feature mapping configurations, feature scaling arrays, and tracking structures match your training datasets exactly.*
    """)

with tab1:
    st.subheader("📋 Patient Information Intake")
    
    # Organized columns with a generous gap layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🧑 Demographics & Symptoms")
        age = st.number_input("Age (Years)", min_value=1, max_value=110, value=45)
        sex = st.selectbox("Gender", options=["Male", "Female"])
        
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain = st.selectbox("Chest Pain Type", options=cp_options, 
                                  help="Typical: classic radiating chest pain. Atypical: non-classical pain. Non-anginal: spasm/sharp pains. Asymptomatic: silent indicators.")
        
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"],
                             help="Does rigorous activity trigger chest/anginal pain pathways?")

    with col2:
        st.markdown("#### 🩺 Physiological & Lab Metrics")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=650, value=200)
        thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])
        
        restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        restecg = st.selectbox("Resting ECG Results", options=restecg_options)
        
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        st_slope_options = ["Up", "Flat", "Down"]
        st_slope = st.selectbox("ST Slope", options=st_slope_options, help="Slope behavior recorded at peak exercise exertion.")

    st.markdown("---")

    # Feature Mapping & Encoding Logics
    # Realignment: Setting Male to 1, Female to 0 to align safely with your comment
    gender_encoded = 1 if sex == "Male" else 0
    fbs_encoded = 1 if fbs == "True" else 0
    exang_encoded = 1 if exang == "Yes" else 0

    # ChestPainType One-Hot Mapping
    asy = 1 if chest_pain == "Asymptomatic" else 0
    ata = 1 if chest_pain == "Atypical Angina" else 0
    nap = 1 if chest_pain == "Non-anginal Pain" else 0
    ta = 1 if chest_pain == "Typical Angina" else 0

    # RestingECG One-Hot Mapping
    lvh = 1 if restecg == "Left Ventricular Hypertrophy" else 0
    normal = 1 if restecg == "Normal" else 0
    St = 1 if restecg == "ST-T Wave Abnormality" else 0

    # ST_Slope One-Hot Mapping
    down = 1 if st_slope == "Down" else 0
    up = 1 if st_slope == "Up" else 0
    flat = 1 if st_slope == "Flat" else 0

    # 2. Reconstruct DataFrame matching exact training list
    data_dict = {
        'Age': age, 
        'Gender': gender_encoded, 
        'RestingBP': trestbps, 
        'Cholesterol': chol, 
        'FastingBS': fbs_encoded, 
        'MaxHR': thalach, 
        'ExerciseAngina': exang_encoded, 
        'Oldpeak': oldpeak, 
        'ChestPainType_ASY': asy, 
        'ChestPainType_ATA': ata, 
        'ChestPainType_NAP': nap, 
        'ChestPainType_TA': ta, 
        'RestingECG_LVH': lvh, 
        'RestingECG_Normal': normal, 
        'RestingECG_ST': St, 
        'ST_Slope_Down': down, 
        'ST_Slope_Flat': flat, 
        'ST_Slope_Up': up
    }

    input_data = pd.DataFrame([data_dict])

    # Dynamic column ordering verification
    input_data = input_data[columns]

    # Action layout configurations
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run_prediction = st.button("🔮 Run Diagnostic Prediction", type="primary", use_container_width=True)

    if run_prediction:
        # Pipeline transformations
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)[0][1]
        
        st.markdown("### 🎯 Diagnostic Report Summary")
        
        # Interactive response metrics framework
        res_col1, res_col2 = st.columns([1, 2], gap="large")
        
        with res_col1:
            if prediction[0] == 1:
                st.metric(label="Calculated Risk Category", value="HIGH RISK", delta="Positive Screen", delta_color="inverse")
            else:
                st.metric(label="Calculated Risk Category", value="LOW RISK", delta="Negative Screen", delta_color="normal")
                
        with res_col2:
            st.write("**Model Risk Probability Tracking Meter:**")
            st.progress(float(prediction_proba))
            st.write(f"Calculated Probability Value: **{prediction_proba:.2%}**")

        # Visual messaging breakouts
        if prediction[0] == 1:
            st.error(f"⚠️ **Clinical Notice:** The underlying algorithm flagged a significant heart disease risk calculation profile (**{prediction_proba:.2%}** probability). Comprehensive imaging or clinical follow-up is advisable.")
        else:
            st.success(f"✅ **Clinical Notice:** The underlying algorithm determined a low overall risk profile (**{1 - prediction_proba:.2%}** confidence score). No clinical anomalies detected based on active tracking variables.")

st.markdown("---")
st.caption("**Disclaimer:** This software tool is engineered for informational, educational, and computational demonstration purposes. It does not construct medical diagnoses, treatment paths, or replace qualified healthcare provider screenings.")