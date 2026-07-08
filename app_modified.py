import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ------------------------------------------------------------------
# 1. Load your artifacts safely
# ------------------------------------------------------------------
model = joblib.load("Log_Heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
columns = joblib.load("columns_heart.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ------------------------------------------------------------------
# 2. Compute the prediction FIRST (real-time, before any styling)
#    so the background color can react to the current risk level.
# ------------------------------------------------------------------
def encode_inputs(age, sex, chest_pain, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, st_slope):
    gender_encoded = 1 if sex == "Female" else 0
    fbs_encoded = 1 if fbs == "True" else 0
    exang_encoded = 1 if exang == "Yes" else 0

    asy = 1 if chest_pain == "Asymptomatic" else 0
    ata = 1 if chest_pain == "Atypical Angina" else 0
    nap = 1 if chest_pain == "Non-anginal Pain" else 0
    ta = 1 if chest_pain == "Typical Angina" else 0

    lvh = 1 if restecg == "Left Ventricular Hypertrophy" else 0
    normal = 1 if restecg == "Normal" else 0
    st_ecg = 1 if restecg == "ST-T Wave Abnormality" else 0

    down = 1 if st_slope == "Down" else 0
    up = 1 if st_slope == "Up" else 0
    flat = 1 if st_slope == "Flat" else 0

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
        'RestingECG_ST': st_ecg,
        'ST_Slope_Down': down,
        'ST_Slope_Flat': flat,
        'ST_Slope_Up': up
    }
    return pd.DataFrame([data_dict])


def risk_theme(proba):
    """Return (label, color, gradient) based on risk probability."""
    if proba is None:
        return "Awaiting input", "#6c757d", ("#f5f7fa", "#e9ecef")
    if proba < 0.35:
        return "Low Risk", "#1f9d55", ("#e6fbf0", "#c8f7dc")
    elif proba < 0.65:
        return "Moderate Risk", "#e0a300", ("#fff8e1", "#ffecb3")
    else:
        return "High Risk", "#d64545", ("#fdecec", "#ffd6d6")


# ------------------------------------------------------------------
# 3. Sidebar inputs (kept live so every change triggers a rerun ->
#    real-time prediction with no button needed)
# ------------------------------------------------------------------
st.sidebar.header("🩺 Patient Metrics")

age = st.sidebar.slider("Age", min_value=1, max_value=110, value=45)
sex = st.sidebar.radio("Gender", options=["Male", "Female"], horizontal=True)

cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
chest_pain = st.sidebar.selectbox("Chest Pain Type", options=cp_options)

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=650, value=200)

fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"], horizontal=True)

restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
restecg = st.sidebar.selectbox("Resting ECG Results", options=restecg_options)

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.radio("Exercise Induced Angina", options=["No", "Yes"], horizontal=True)
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st_slope_options = ["Up", "Down", "Flat"]
st_slope = st.sidebar.selectbox("ST Slope", options=st_slope_options)

live_mode = st.sidebar.toggle("⚡ Real-time prediction", value=True,
                               help="Predicts instantly as you move the sliders. Turn off to use the button instead.")

# ------------------------------------------------------------------
# 4. Build feature frame + run prediction (live, or on button click)
# ------------------------------------------------------------------
input_data = encode_inputs(age, sex, chest_pain, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, st_slope)
input_data = input_data[columns]

proba = None
prediction = None
run_clicked = st.session_state.get("run_clicked", False)

if live_mode:
    scaled_features = scaler.transform(input_data)
    prediction = model.predict(scaled_features)
    proba = model.predict_proba(scaled_features)[0][1]

label, accent, (bg_top, bg_bottom) = risk_theme(proba)

# ------------------------------------------------------------------
# 5. Dynamic background styling — reacts to current risk level
# ------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {bg_top} 0%, {bg_bottom} 100%);
    transition: background 0.6s ease;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #2b2d42 0%, #1a1c2c 100%);
}}
[data-testid="stSidebar"] * {{
    color: #f1f1f1 !important;
}}
/* Widgets that render on a white/light background need dark text,
   otherwise the blanket rule above makes them white-on-white */
[data-testid="stSidebar"] [data-baseweb="select"] * ,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="input"] * {{
    color: #1a1c2c !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    fill: #1a1c2c !important;
}}
div.stButton > button {{
    background-color: {accent};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.15);
}}
.risk-badge {{
    display: inline-block;
    padding: 0.35em 1em;
    border-radius: 999px;
    background-color: {accent};
    color: white;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.03em;
}}
.metric-card {{
    background: white;
    border-radius: 12px;
    padding: 1em;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 6. Header
# ------------------------------------------------------------------
st.title("❤️ Heart Disease Diagnostic Assistant")
st.write("Adjust the patient's clinical metrics in the sidebar. With real-time mode on, "
         "the prediction and gauge update instantly as you move each slider.")
st.markdown(f'<span class="risk-badge">{label}</span>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# 7. Manual run button (used when live mode is off)
# ------------------------------------------------------------------
if not live_mode:
    if st.button("🔮 Run Diagnostic Prediction", type="primary"):
        st.session_state["run_clicked"] = True
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)[0][1]
        label, accent, _ = risk_theme(proba)

# ------------------------------------------------------------------
# 8. Results: gauge meter + summary cards
# ------------------------------------------------------------------
if proba is not None:
    risk_pct = proba * 100

    col_gauge, col_summary = st.columns([1.3, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix': "%", 'font': {'size': 40}},
            title={'text': "Predicted Risk Score", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': accent, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 35], 'color': '#c8f7dc'},
                    {'range': [35, 65], 'color': '#ffecb3'},
                    {'range': [65, 100], 'color': '#ffd6d6'},
                ],
                'threshold': {
                    'line': {'color': accent, 'width': 4},
                    'thickness': 0.85,
                    'value': risk_pct
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if prediction is not None and prediction[0] == 1:
            st.error(f"⚠️ **{label}**\n\nModel predicts presence of heart disease.")
        else:
            st.success(f"✅ **{label}**\n\nModel predicts no significant heart disease.")
        st.metric("Confidence (Disease)", f"{risk_pct:.1f}%")
        st.metric("Confidence (No Disease)", f"{100 - risk_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.progress(min(max(proba, 0.0), 1.0))
else:
    st.info("👈 Adjust the sidebar inputs, or click **Run Diagnostic Prediction** to see results.")

st.markdown("---")
st.info("**Disclaimer:** This tool is for informational/educational purposes only and does not "
        "substitute for a professional medical diagnosis.")