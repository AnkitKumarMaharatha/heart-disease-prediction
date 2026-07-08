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
    """Return (label, neon_color, glow_color) based on risk probability."""
    if proba is None:
        return "Awaiting input", "#00e5ff", "rgba(0, 229, 255, 0.55)"
    if proba < 0.35:
        return "Low Risk", "#39ff14", "rgba(57, 255, 20, 0.55)"
    elif proba < 0.65:
        return "Moderate Risk", "#ffea00", "rgba(255, 234, 0, 0.55)"
    else:
        return "High Risk", "#ff2079", "rgba(255, 32, 121, 0.6)"


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

label, accent, glow = risk_theme(proba)

# ------------------------------------------------------------------
# 5. Dynamic background styling — reacts to current risk level
# ------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

@keyframes soft-glow {{
    0%   {{ box-shadow: 0 0 4px {glow}; }}
    50%  {{ box-shadow: 0 0 14px {glow}; }}
    100% {{ box-shadow: 0 0 4px {glow}; }}
}}

/* ===== App-wide backdrop: clean dark slate, subtle accent glow ===== */
.stApp {{
    background:
        radial-gradient(circle at 10% 0%, {glow} 0%, transparent 35%),
        linear-gradient(160deg, #0f1117 0%, #151824 50%, #0f1117 100%);
    background-attachment: fixed;
    transition: background 0.8s ease;
}}
.stApp, .stApp p, .stApp label, .stApp span, .stMarkdown {{
    color: #e6e9f2;
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}}

/* ===== Sidebar shell ===== */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #171a26 0%, #0f1117 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {{
    color: #e6e9f2 !important;
    font-family: 'Inter', sans-serif;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {accent} !important;
    font-weight: 700 !important;
}}

/* ===== Dropdowns / text inputs =====
   Solid light field + dark text = guaranteed contrast, no matter which
   inner element Streamlit/BaseWeb renders the value in. */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] input {{
    background-color: #f4f6fb !important;
    border: 1.5px solid {accent} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] div[data-baseweb="select"] > div *,
[data-testid="stSidebar"] div[data-baseweb="input"] > div *,
[data-testid="stSidebar"] input {{
    color: #10121a !important;
    -webkit-text-fill-color: #10121a !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
    fill: #10121a !important;
}}
/* Dropdown option list is portalled outside the sidebar; style it globally */
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {{
    background-color: #f4f6fb !important;
    color: #10121a !important;
}}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {{
    background-color: #e2e7f5 !important;
}}

/* ===== Sliders ===== */
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {{
    background-color: {accent} !important;
    box-shadow: 0 0 8px {glow};
    border: 2px solid white;
}}

/* ===== Buttons ===== */
div.stButton > button {{
    background: {accent};
    color: #0f1117;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.4em;
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 0 16px {glow};
}}

/* ===== Risk badge ===== */
.risk-badge {{
    display: inline-block;
    padding: 0.4em 1.2em;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid {accent};
    color: {accent} !important;
    font-weight: 700;
    font-size: 0.85em;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    animation: soft-glow 3s ease-in-out infinite;
}}

/* ===== Result cards / metrics ===== */
.metric-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2em;
    text-align: center;
}}
[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.6em;
}}
[data-testid="stMetricValue"] {{
    color: {accent} !important;
}}

/* Progress bar */
[data-testid="stProgress"] > div > div {{
    background: {accent} !important;
}}

/* Alerts */
[data-testid="stAlert"] {{
    background-color: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px;
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
        label, accent, glow = risk_theme(proba)

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
            number={'suffix': "%", 'font': {'size': 42, 'color': accent, 'family': "Inter"}},
            title={'text': "PREDICTED RISK SCORE",
                   'font': {'size': 14, 'color': '#e6e9f2', 'family': "Inter"}},
            gauge={
                'shape': "angular",
                'axis': {'range': [0, 100], 'tickwidth': 1,
                         'tickcolor': '#e6e9f2', 'tickfont': {'color': '#e6e9f2'}},
                'bar': {'color': accent, 'thickness': 0.28},
                'bgcolor': "#05060f",
                'borderwidth': 1,
                'bordercolor': accent,
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(57, 255, 20, 0.25)'},
                    {'range': [35, 65], 'color': 'rgba(255, 234, 0, 0.25)'},
                    {'range': [65, 100], 'color': 'rgba(255, 32, 121, 0.25)'},
                ],
                'threshold': {
                    'line': {'color': accent, 'width': 5},
                    'thickness': 0.9,
                    'value': risk_pct
                }
            }
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#e6e9f2'}
        )
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