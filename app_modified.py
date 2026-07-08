import streamlit as st
import pandas as pd
import joblib

# 1. Load your artifacts safely
model = joblib.load("Log_Heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
columns = joblib.load("columns_heart.pkl")

st.set_page_config(
    page_title="Heart Disease Analytics Engine",
    page_icon="❤️",
    layout="wide"
)

# FORCE-INJECT CSS: This overrides Streamlit's internal block containers directly
st.markdown("""
    <style>
    /* 1. Target the FIRST input container block (Demographics) */
    div[data-testid="stVerticalBlock"] > div:nth-child(2) > div[data-testid="stVerticalBlockBorder"] {
        background-color: #EBF8FF !important; /* Soft Blue */
        border: 2px solid #3182CE !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 2. Target the SECOND input container block (Labs) */
    div[data-testid="stVerticalBlock"] > div:nth-child(3) > div[data-testid="stVerticalBlockBorder"] {
        background-color: #F7FAFC !important; /* Cool Grey */
        border: 2px solid #CBD5E0 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
    }

    /* Dark Mode specific overrides so text stays crisp and readable */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stVerticalBlock"] > div:nth-child(2) > div[data-testid="stVerticalBlockBorder"] {
            background-color: #1A365D !important; /* Dark Navy Blue */
            border-color: #2B6CB0 !important;
        }
        div[data-testid="stVerticalBlock"] > div:nth-child(3) > div[data-testid="stVerticalBlockBorder"] {
            background-color: #2D3748 !important; /* Dark Slate Grey */
            border-color: #4A5568 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Advanced Heart Disease Analytics Engine")
st.write("Modify patient inputs in Tab 1 and switch to Tab 2 for real-time tracking.")
st.markdown("---")

tab1, tab2 = st.tabs(["📋 [Step 1] Patient Metric Input", "🎯 [Step 2] Real-Time Prediction Analytics"])

with tab1:
    # First Styled Block: Demographics
    with st.container(border=True):
        st.markdown("### 🧑‍⚕️ Patient Demographics & Symptom Profile")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            age = st.number_input("Age (Years)", min_value=1, max_value=110, value=45)
            sex = st.selectbox("Gender Identification", options=["Male", "Female"])
            cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            chest_pain = st.selectbox("Chest Pain Categorization", options=cp_options)
            exang = st.selectbox("Exercise Induced Angina Symptoms", options=["No", "Yes"])
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Serum Cholesterol Level (mg/dl)", min_value=0, max_value=650, value=200)
            thalach = st.number_input("Maximum Achieved Heart Rate (bpm)", min_value=60, max_value=220, value=150)

    # Second Styled Block: Electrocardiogram
    with st.container(border=True):
        st.markdown("### 🔬 Advanced Electrocardiogram & Lab Metrics")
        col3, col4 = st.columns(2, gap="large")
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar Profile > 120 mg/dl", options=["False", "True"])
            restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            restecg = st.selectbox("Resting ECG Waveform Analysis", options=restecg_options)
        with col4:
            oldpeak = st.number_input("ST Depression Induced via Physical Stress", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            st_slope_options = ["Up", "Flat", "Down"]
            st_slope = st.selectbox("Peak Exercise ST Segment Slope", options=st_slope_options)

    st.info("💡 **Data Live Synced:** Head over to the **Real-Time Prediction Analytics** tab to view results instantly.")

# Behind-the-scenes data engineering pipeline
gender_encoded = 1 if sex == "Male" else 0
fbs_encoded = 1 if fbs == "True" else 0
exang_encoded = 1 if exang == "Yes" else 0

asy = 1 if chest_pain == "Asymptomatic" else 0
ata = 1 if chest_pain == "Atypical Angina" else 0
nap = 1 if chest_pain == "Non-anginal Pain" else 0
ta = 1 if chest_pain == "Typical Angina" else 0

lvh = 1 if restecg == "Left Ventricular Hypertrophy" else 0
normal = 1 if restecg == "Normal" else 0
St = 1 if restecg == "ST-T Wave Abnormality" else 0

down = 1 if st_slope == "Down" else 0
up = 1 if st_slope == "Up" else 0
flat = 1 if st_slope == "Flat" else 0

data_dict = {
    'Age': age, 'Gender': gender_encoded, 'RestingBP': trestbps, 'Cholesterol': chol, 
    'FastingBS': fbs_encoded, 'MaxHR': thalach, 'ExerciseAngina': exang_encoded, 'Oldpeak': oldpeak, 
    'ChestPainType_ASY': asy, 'ChestPainType_ATA': ata, 'ChestPainType_NAP': nap, 'ChestPainType_TA': ta, 
    'RestingECG_LVH': lvh, 'RestingECG_Normal': normal, 'RestingECG_ST': St, 
    'ST_Slope_Down': down, 'ST_Slope_Flat': flat, 'ST_Slope_Up': up
}

input_data = pd.DataFrame([data_dict])[columns]

# Machine Learning Calculations
scaled_features = scaler.transform(input_data)
prediction = model.predict(scaled_features)
prediction_proba = float(model.predict_proba(scaled_features)[0][1])

with tab2:
    st.subheader("🎯 Live Clinical Diagnostic Report")
    
    # Real-Time colored status blocks using native Streamlit notification alerts (which are completely filled)
    if prediction[0] == 1:
        st.error(f"## ⚠️ Alert: High Probability of Anomalies Detected ({prediction_proba:.2%})")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(label="Evaluated Risk Index", value="HIGH RISK", delta="Positive Finding", delta_color="inverse")
        with c2:
            st.write("**Active Probability Risk Threshold Status:**")
            st.progress(prediction_proba)
            
    else:
        st.success(f"## ✅ Clear Status: Standard Metrics Registered ({1 - prediction_proba:.2%})")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(label="Evaluated Risk Index", value="LOW RISK", delta="Negative Finding", delta_color="normal")
        with c2:
            st.write("**Active Probability Risk Threshold Status:**")
            st.progress(prediction_proba)

    with st.expander("🔍 View Raw Transformed Vector Metrics", expanded=False):
        st.dataframe(input_data, use_container_width=True)

st.markdown("---")
st.caption("**Disclaimer:** This software tool is engineered for informational, educational, and computational demonstration purposes.")