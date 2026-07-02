import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go  # Added for interactive gauge chart

# 1. Page & Styling Configuration
st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="centered"
)

# Custom Cyberpunk/Dark Mode CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050505; 
        color: #00F0FF; 
    }
    [data-testid="stSidebar"] {
        background-color: #0D0D11;
        border-right: 1px solid #FF0055;
    }
    label {
        color: #FF0055 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }
    /* Clean metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Load artifacts safely
@st.cache_resource
def load_assets():
    # Fallback simulation if files aren't found locally for testing
    try:
        model = joblib.load("Log_Heart.pkl")
        columns = joblib.load("columns_heart.pkl")
        scaler = joblib.load("scaler_heart.pkl")
        return model, columns, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, columns, scaler = load_assets()

# 3. UI Header
st.title("🫀 CardioRisk: Intelligent Diagnostic Assistant")
st.write("Modify patient demographics and vital clinical metrics below. The diagnostic engine dynamically visualizes patient cardiovascular profiles instantly.")
st.markdown("---")

# 4. Grouping Inputs into Interactive Tabs
tab1, tab2, tab3 = st.tabs(["👤 Patient Profile", "📊 Vitals & Labs", "📈 Interactive Analysis"])

with tab1:
    st.subheader("Demographics & Symptoms")
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        age = st.slider("Age (Years)", min_value=1, max_value=110, value=45, help="Select the patient's age.")
        sex = st.radio("Gender", options=["Male", "Female"], horizontal=True)
    
    with col1_2:
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain = st.selectbox("Chest Pain Type", options=cp_options, 
                                help="Type of chest pain experienced by the patient.")
        exang = st.radio("Exercise Induced Angina", options=["No", "Yes"], horizontal=True,
                           help="Does exercise trigger chest pain/angina symptoms?")

with tab2:
    st.subheader("Cardiovascular Test Results")
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
        # Interactive warning trigger
        if trestbps >= 140:
            st.warning("⚠️ Hypertension Threshold: Blood pressure exceeds 140 mm Hg.")
            
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, 
                         help="Some medical datasets record 0 for missing values.")
        if chol > 240:
            st.warning("⚠️ Hypercholesterolemia Threshold: Cholesterol exceeds 240 mg/dl.")
            
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])

    with col2_2:
        thalach = st.slider("Maximum Heart Rate Achieved (MaxHR)", min_value=60, max_value=220, value=150)
        oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak)", min_value=-3.0, max_value=10.0, value=0.0, step=0.1,
                            help="ST depression relative to rest. Higher values indicate higher cardiovascular strain.")
        
        st_slope_options = ["Up", "Flat", "Down"]
        st_slope = st.selectbox("ST Slope Type", options=st_slope_options)
        
        restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        restecg = st.selectbox("Resting ECG Results", options=restecg_options)

# 5. Data Mapping
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

ordered_features = [
    age, gender_encoded, trestbps, chol, fbs_encoded, thalach, exang_encoded, oldpeak,
    asy, ata, nap, ta, lvh, normal, St, down, flat, up
]

input_data = pd.DataFrame([ordered_features], columns=columns)
scaled = scaler.transform(input_data)

# Run Inference
prediction = model.predict(scaled)
prediction_proba = model.predict_proba(scaled)[0][1]

# Fill tab3 with advanced interactive outputs
with tab3:
    st.subheader("📊 Dynamic Diagnostic Risk Assessment")
    
    # 1. Create Plotly Gauge Chart for dynamic visualization
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk Probability (%)", 'font': {'color': '#00F0FF', 'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00F0FF"},
            'bar': {'color': "#FF0055" if prediction[0] == 1 else "#00FF66"},
            'bgcolor': "#111111",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 102, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(255, 200, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 85, 0.1)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00F0FF", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, width='stretch')

    # 2. Status Output Layout
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        if prediction[0] == 1:
            st.metric(label="Calculated Classification", value="HIGH RISK", delta="Action Advised", delta_color="inverse")
        else:
            st.metric(label="Calculated Classification", value="LOW RISK", delta="Normal Profile", delta_color="normal")

    with metric_col2:
        # Dynamic advice rendering based on clinical numbers
        st.markdown("**Core Findings Summary:**")
        if prediction[0] == 1:
            st.write(f"Patient falls into a statistically higher vulnerability category. The model flags a `{prediction_proba:.1%}` match with legacy ischemic profiles.")
        else:
            st.write(f"The baseline attributes showcase healthy tracking thresholds. Risk margin stays securely low at `{prediction_proba:.1%}`.")

    # 3. Dynamic Patient Summary Report Generator
    st.markdown("---")
    st.subheader("📋 Auto-Generated Clinical Summary")
    
    summary_text = f"""
    **Patient Medical Profile Summary**
    *   **Age/Gender:** {age} Year Old {sex}
    *   **Symptom Type:** Presenting with {chest_pain} chest discomfort.
    *   **Vitals Tracking:** Blood pressure registered at {trestbps} mm Hg with a serum lipid panel holding at {chol} mg/dl.
    *   **Diagnostic Vector Result:** Model evaluates standard classification status at **{"HIGH RISK" if prediction[0]==1 else "LOW RISK"}**.
    """
    st.info(summary_text)

# Keep standard footer items outside of tabs
st.markdown("---")
with st.expander("🛠️ View Raw Model Feature Matrix"):
    st.write("This is the exact synchronized 1x18 matrix passed to the Logistic Regression model:")
    st.dataframe(input_data)

st.info("**Disclaimer:** This application serves purely as an educational/portfolio demonstration of a machine learning classifier. It does not replace clinical consultation.")