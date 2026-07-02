import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page & Styling Configuration
st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="centered"
)

# Custom CSS for a clean modern look
# Custom CSS for a clean modern look
# Custom CSS forcing a light gray background color
# Custom CSS forcing a light gray background color
# Custom CSS forcing a light gray background color safely across all viewport caches
st.markdown(
    """
    <style>
    /* Clean Light Background */
    .stApp {
        background-color: #F8FAFC; 
        color: #1E293B;
    }
    /* Soft Blue-Gray Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E2E8F0;
    }
    /* Crisp dark gray text for labels */
    label {
        color: #334155 !important;
        font-weight: 600;
    }
    /* Soft Red alert for High Risk */
    .prediction-box-high {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        padding: 20px;
        border-radius: 10px;
        color: #991B1B;
        text-align: center;
    }
    /* Soft Green alert for Low Risk */
    .prediction-box-safe {
        background-color: #DCFCE7;
        border: 2px solid #22C55E;
        padding: 20px;
        border-radius: 10px;
        color: #166534;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Load artifacts safely
@st.cache_resource
def load_assets():
    model = joblib.load("Log_Heart.pkl")
    columns = joblib.load("columns_heart.pkl")
    scaler = joblib.load("scaler_heart.pkl")
    return model, columns, scaler

try:
    model, columns, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. UI Header
st.title("🫀 CardioRisk: Real-time Diagnostic Assistant")
st.write("Adjust the clinical parameters below. The AI model will recalculate heart disease probability instantly.")
st.markdown("---")

# 4. Grouping Inputs into Interactive Tabs
tab1, tab2 = st.tabs(["👤 Patient Demographics", "📊 Clinical & ECG Metrics"])

with tab1:
    st.subheader("Basic Patient Information")
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        age = st.slider("Age (Years)", min_value=1, max_value=110, value=45, help="Select the patient's age.")
        sex = st.selectbox("Gender", options=["Male", "Female"])
    
    with col1_2:
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain = st.selectbox("Chest Pain Type", options=cp_options, 
                                help="Type of chest pain experienced by the patient.")
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"],
                             help="Does exercise cause chest pain/angina?")

with tab2:
    st.subheader("Cardiovascular Test Results")
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, 
                         help="Some medical datasets record 0 for missing values.")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])

    with col2_2:
        thalach = st.slider("Maximum Heart Rate Achieved (MaxHR)", min_value=60, max_value=220, value=150)
        oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak)", min_value=-3.0, max_value=10.0, value=0.0, step=0.1,
                            help="ST depression relative to rest. Higher numeric values generally correlate with higher cardiovascular risk.")
        
        st_slope_options = ["Up", "Flat", "Down"]
        st_slope = st.selectbox("ST Slope Type", options=st_slope_options)
        
        restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        restecg = st.selectbox("Resting ECG Results", options=restecg_options)

st.markdown("---")

# 5. Exact Dataset Categorical Alignment Mapping
gender_encoded = 1 if sex == "Female" else 0 
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
flat = 1 if st_slope == "Flat" else 0
up = 1 if st_slope == "Up" else 0

# 6. Reconstruct ordered features list based on the extracted sequence array
ordered_features = [
    age, gender_encoded, trestbps, chol, fbs_encoded, thalach, exang_encoded, oldpeak,
    asy, ata, nap, ta, lvh, normal, St, down, flat, up
]

# Convert it into a 2D dataframe mapping exactly to columns artifact array
input_data = pd.DataFrame([ordered_features], columns=columns)

scaled = scaler.transform(input_data)

# 7. REAL-TIME PREDICTION SECTION (No button needed!)
prediction = model.predict(scaled)
prediction_proba = model.predict_proba(scaled)[0][1]

# Displaying dynamic results beautifully
st.subheader("📊 Diagnostic Risk Assessment")

metric_col1, metric_col2 = st.columns([1, 2])

with metric_col1:
    if prediction[0] == 1:
        st.metric(label="Calculated Status", value="HIGH RISK", delta="Action Advised", delta_color="inverse")
    else:
        st.metric(label="Calculated Status", value="LOW RISK", delta="Normal", delta_color="normal")

with metric_col2:
    st.write(f"**AI Model Confidence Score:** {prediction_proba:.1%}")
    # Colorful progress bar tracking risk percentage
    st.progress(int(prediction_proba * 100))

# Informational alert box
if prediction[0] == 1:
    st.error(f"⚠️ High Probability: The model indicates a **{prediction_proba:.1%}** likelihood of heart disease presence based on the clinical profile.")
else:
    st.success(f"✅ Low Probability: The model indicates a **{1 - prediction_proba:.1%}** likelihood of absence of heart disease.")

# Expandable Data Debugger
with st.expander("🛠️ View Raw Model Feature Matrix"):
    st.write("This is the exact synchronized 1x18 matrix passed to the Logistic Regression model:")
    st.dataframe(input_data)

st.info("**Disclaimer:** This application serves purely as an educational/portfolio demonstration of a machine learning classifier. It does not replace clinical consultation.")