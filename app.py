import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="CardioPredict | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #FAFBFF;
        color: #0D1B2A;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding: 0rem 2.5rem 2rem 2.5rem; max-width: 1280px; }

    /* ── TOP NAV BAR ── */
    .topnav {
        background: white;
        border-bottom: 1px solid #E8EDF5;
        padding: 14px 0 12px 0;
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .topnav-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 22px;
        color: #1A56DB;
        letter-spacing: -0.3px;
    }
    .topnav-tag {
        background: #EEF2FF;
        color: #4F46E5;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    }

    /* ── HERO SECTION ── */
    .hero {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 20px;
        padding: 48px 56px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 340px; height: 100%;
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        clip-path: ellipse(100% 100% at 100% 50%);
    }
    .hero-eyebrow {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4F46E5;
        margin-bottom: 12px;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 44px;
        line-height: 1.15;
        color: #0D1B2A;
        margin-bottom: 14px;
        position: relative;
        z-index: 1;
    }
    .hero-title span { color: #1A56DB; }
    .hero-desc {
        font-size: 16px;
        color: #4B5563;
        line-height: 1.7;
        max-width: 560px;
        position: relative;
        z-index: 1;
        margin-bottom: 24px;
    }
    .hero-stats {
        display: flex;
        gap: 32px;
        position: relative;
        z-index: 1;
    }
    .hero-stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        color: #1A56DB;
        line-height: 1;
    }
    .hero-stat-label {
        font-size: 12px;
        color: #6B7280;
        margin-top: 3px;
    }

    /* ── STAT CARDS ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 28px;
    }
    .stat-card {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 14px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
    }
    .stat-card.blue::after  { background: linear-gradient(90deg, #1A56DB, #60A5FA); }
    .stat-card.red::after   { background: linear-gradient(90deg, #DC2626, #F87171); }
    .stat-card.green::after { background: linear-gradient(90deg, #059669, #34D399); }
    .stat-card.purple::after{ background: linear-gradient(90deg, #7C3AED, #A78BFA); }
    .stat-icon {
        font-size: 24px;
        margin-bottom: 10px;
        display: block;
    }
    .stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 32px;
        line-height: 1;
        color: #0D1B2A;
    }
    .stat-label {
        font-size: 12px;
        color: #6B7280;
        margin-top: 4px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 12px;
        padding: 5px;
        gap: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 10px 22px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1A56DB, #4F46E5) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(26,86,219,0.3) !important;
    }

    /* ── FORM SECTIONS ── */
    .form-section {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 16px;
        padding: 28px;
        height: 100%;
    }
    .form-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        padding-bottom: 14px;
        border-bottom: 1px solid #F3F4F6;
    }
    .form-section-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .icon-blue   { background: #EEF2FF; }
    .icon-teal   { background: #F0FDFA; }
    .icon-amber  { background: #FFFBEB; }
    .form-section-title {
        font-size: 14px;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.1px;
    }
    .form-section-sub {
        font-size: 11px;
        color: #9CA3AF;
        margin-top: 1px;
    }

    /* ── INPUT HINT ── */
    .hint-bar {
        background: #F5F7FF;
        border: 1px dashed #C7D2FE;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        color: #4338CA;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── PREDICT BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #1A56DB 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(26,86,219,0.3) !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(26,86,219,0.4) !important;
    }

    /* ── RESULT CARDS ── */
    .result-wrap {
        border-radius: 16px;
        padding: 28px 32px;
        margin: 20px 0;
        border: 1px solid;
    }
    .result-disease {
        background: #FFF5F5;
        border-color: #FEB2B2;
    }
    .result-healthy {
        background: #F0FFF4;
        border-color: #9AE6B4;
    }
    .result-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .badge-danger  { background:#FED7D7; color:#C53030; }
    .badge-success { background:#C6F6D5; color:#276749; }
    .result-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        margin-bottom: 6px;
    }
    .result-disease .result-heading { color: #C53030; }
    .result-healthy .result-heading { color: #276749; }
    .result-desc { font-size: 14px; color: #4A5568; line-height: 1.6; }

    /* ── SUMMARY TABLE ── */
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 16px;
    }
    .summary-item {
        background: #F9FAFB;
        border: 1px solid #F3F4F6;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .summary-key   { font-size: 11px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px; }
    .summary-value { font-size: 15px; font-weight: 600; color: #111827; }

    /* ── SECTION LABEL ── */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9CA3AF;
        margin: 28px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #F3F4F6;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #0D1B2A !important;
    }
    [data-testid="stSidebar"] * { color: #E5E7EB !important; }
    .sidebar-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        color: white !important;
        text-align: center;
        padding: 24px 0 8px 0;
    }
    .sidebar-tagline {
        font-size: 12px;
        color: #6B7280 !important;
        text-align: center;
        margin-bottom: 28px;
    }
    .sidebar-divider {
        height: 1px;
        background: #1F2937;
        margin: 16px 0;
    }
    .sidebar-section-title {
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: #4B5563 !important;
        margin-bottom: 10px !important;
    }
    .sidebar-item {
        font-size: 13px !important;
        color: #9CA3AF !important;
        padding: 4px 0 !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── FOOTER ── */
    .app-footer {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 14px;
        padding: 22px 32px;
        text-align: center;
        margin-top: 48px;
        color: #6B7280;
        font-size: 13px;
    }
    .app-footer a { color: #1A56DB; text-decoration: none; font-weight: 500; }
    .app-footer strong { color: #0D1B2A; }

    /* Streamlit slider & select */
    .stSlider label, .stSelectbox label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('dataset_heart.csv')
    df.columns = df.columns.str.strip()
    if 'heart disease' in df.columns:
        df = df.rename(columns={'heart disease': 'target'})
        df['target'] = df['target'].map({1: 0, 2: 1})
    return df

model, scaler = load_model()
df = load_data()

total   = len(df)
disease = int(df['target'].sum())
healthy = int((df['target'] == 0).sum())

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🫀 CardioPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Clinical Decision Support System</div>', unsafe_allow_html=True)
    # st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # st.markdown('<div class="sidebar-section-title">About</div>', unsafe_allow_html=True)
    # st.markdown("""<div style='font-size:13px; color:#6B7280; line-height:1.75;'>
    # CardioPredict uses a trained ML classifier to assess cardiovascular risk 
    # based on 13 clinical indicators from the UCI Heart Disease dataset.</div>""",
    # unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Dataset Info</div>', unsafe_allow_html=True)
    for item in [f"📊 {total} patient records", "🧬 13 clinical features",
                 "✅ No missing values", "📅 UCI Heart Disease"]:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Author</div>', unsafe_allow_html=True)
    st.markdown("""<div style='font-size:13px; color:#6B7280; line-height:1.9;'>
    <b style='color:#E5E7EB'>M U F Ushna</b><br>
    IT23436312 · SLIIT<br>
    <a href='https://github.com/Ushna001/Heart-Disease-Prediction-ML'
       style='color:#60A5FA;'>GitHub →</a></div>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning · Clinical AI · Cardiology</div>
    <div class="hero-title">Predict <span>Heart Disease</span><br>with Clinical Precision</div>
    <div class="hero-desc">
        Enter patient clinical data to receive an instant cardiovascular risk 
        assessment powered by a trained machine learning classifier.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-num">{total}</div>
            <div class="hero-stat-label">Training Records</div>
        </div>
        <div>
            <div class="hero-stat-num">13</div>
            <div class="hero-stat-label">Clinical Features</div>
        </div>
        <div>
            <div class="hero-stat-num">85%</div>
            <div class="hero-stat-label">Model Accuracy</div>
        </div>
        <div>
            <div class="hero-stat-num">92%</div>
            <div class="hero-stat-label">AUC Score</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS ───────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card blue">
        <span class="stat-icon">🏥</span>
        <div class="stat-num">{total}</div>
        <div class="stat-label">Total Patients</div>
    </div>
    <div class="stat-card red">
        <span class="stat-icon">⚠️</span>
        <div class="stat-num" style="color:#DC2626">{disease}</div>
        <div class="stat-label">Heart Disease</div>
    </div>
    <div class="stat-card green">
        <span class="stat-icon">💚</span>
        <div class="stat-num" style="color:#059669">{healthy}</div>
        <div class="stat-label">Healthy</div>
    </div>
    <div class="stat-card purple">
        <span class="stat-icon">🎯</span>
        <div class="stat-num" style="color:#7C3AED">85%</div>
        <div class="stat-label">Best Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Prediction",
    "📊  Data Analysis",
    "🤖  Model Performance",
    "📋  Dataset"
])

# ═══════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Patient Clinical Input</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hint-bar">
        💡 Fill in all clinical fields below. Values are pre-set to dataset averages as defaults.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="form-section-header">
            <div class="form-section-icon icon-blue">👤</div>
            <div>
                <div class="form-section-title">Demographics</div>
                <div class="form-section-sub">Basic patient info</div>
            </div>
        </div>""", unsafe_allow_html=True)
        age = st.slider("Age (years)", 20, 80, 54)
        sex = st.selectbox("Biological Sex", [0,1],
                           format_func=lambda x: "Female" if x==0 else "Male")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1],
                                  format_func=lambda x: "No" if x==0 else "Yes")
        exercise_angina = st.selectbox("Exercise Induced Angina", [0,1],
                                       format_func=lambda x: "No" if x==0 else "Yes")

    with col2:
        st.markdown("""
        <div class="form-section-header">
            <div class="form-section-icon icon-teal">🩺</div>
            <div>
                <div class="form-section-title">Measurements</div>
                <div class="form-section-sub">Vitals & lab results</div>
            </div>
        </div>""", unsafe_allow_html=True)
        resting_bp  = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 131)
        cholesterol = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 249)
        max_hr      = st.slider("Max Heart Rate Achieved (bpm)", 70, 210, 149)
        oldpeak     = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.05, step=0.1)

    with col3:
        st.markdown("""
        <div class="form-section-header">
            <div class="form-section-icon icon-amber">📈</div>
            <div>
                <div class="form-section-title">Diagnostics</div>
                <div class="form-section-sub">ECG & imaging findings</div>
            </div>
        </div>""", unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain Type", [1,2,3,4],
                                  format_func=lambda x: {1:"Typical Angina",2:"Atypical Angina",
                                                          3:"Non-Anginal Pain",4:"Asymptomatic"}[x])
        rest_ecg = st.selectbox("Resting ECG Results", [0,1,2],
                                format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",
                                                        2:"LV Hypertrophy"}[x])
        st_segment    = st.selectbox("ST Segment Slope", [1,2,3],
                                     format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
        major_vessels = st.selectbox("Major Vessels Colored (0–3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia Type", [3,6,7],
                            format_func=lambda x: {3:"Normal",6:"Fixed Defect",7:"Reversible Defect"}[x])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Run Cardiovascular Risk Assessment"):
        inp = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                         fasting_bs, rest_ecg, max_hr, exercise_angina,
                         oldpeak, st_segment, major_vessels, thal]])
        inp_scaled  = scaler.transform(inp)
        prediction  = model.predict(inp_scaled)[0]
        probability = model.predict_proba(inp_scaled)[0]

        st.markdown('<div class="section-label">Risk Assessment Results</div>', unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-wrap result-disease">
                    <span class="result-badge badge-danger">⚠️ High Risk</span>
                    <div class="result-heading">Heart Disease Detected</div>
                    <div class="result-desc">
                        The model predicts a <strong>{probability[1]*100:.1f}%</strong> probability 
                        of heart disease. Please refer the patient to a cardiologist for 
                        further diagnostic evaluation and treatment planning.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-wrap result-healthy">
                    <span class="result-badge badge-success">✅ Low Risk</span>
                    <div class="result-heading">No Disease Detected</div>
                    <div class="result-desc">
                        The model predicts a <strong>{probability[0]*100:.1f}%</strong> probability 
                        of being healthy. Clinical indicators are within acceptable range. 
                        Routine follow-up is recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Healthy Probability",   f"{probability[0]*100:.1f}%")
            c2.metric("Disease Probability",   f"{probability[1]*100:.1f}%")

        with gauge_col:
            color = "#DC2626" if prediction == 1 else "#059669"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability[1]*100, 1),
                title={'text': "Risk Score", 'font': {'size':15,'color':'#374151','family':'DM Sans'}},
                number={'suffix':"%",'font':{'size':42,'color':color,'family':'DM Sans'}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#D1D5DB','tickfont':{'size':11}},
                    'bar':{'color':color,'thickness':0.22},
                    'bgcolor':'#F9FAFB','borderwidth':0,
                    'steps':[
                        {'range':[0,35],'color':'#ECFDF5'},
                        {'range':[35,65],'color':'#FEFCE8'},
                        {'range':[65,100],'color':'#FEF2F2'}
                    ],
                    'threshold':{'line':{'color':'#1A56DB','width':3},
                                 'thickness':0.8,'value':probability[1]*100}
                }
            ))
            fig.update_layout(height=290, paper_bgcolor='white',
                              margin=dict(t=60,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Patient Summary</div>', unsafe_allow_html=True)
        items = [
            ("Age", f"{age} years"),
            ("Sex", "Male" if sex==1 else "Female"),
            ("Chest Pain", {1:"Typical Angina",2:"Atypical Angina",
                            3:"Non-Anginal Pain",4:"Asymptomatic"}[chest_pain]),
            ("Resting BP", f"{resting_bp} mmHg"),
            ("Cholesterol", f"{cholesterol} mg/dl"),
            ("Max Heart Rate", f"{max_hr} bpm"),
            ("ST Depression", str(oldpeak)),
            ("Major Vessels", str(major_vessels)),
        ]
        cols = st.columns(4)
        for i, (k, v) in enumerate(items):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="summary-item">
                    <div class="summary-key">{k}</div>
                    <div class="summary-value">{v}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    for title, path in {
        "Target Distribution":  "target_distribution.png",
        "Correlation Heatmap":  "correlation_heatmap.png",
        "EDA Analysis":         "eda_analysis.png",
        "Feature Importance":   "feature_importance.png"
    }.items():
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(Image.open(path), width="stretch")
            st.markdown("---")
        else:
            st.warning(f"⚠️ {path} not found in folder")

# ═══════════════════════════════════════════
# TAB 3 — MODEL
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    for title, path in {
        "Model Comparison": "model_comparison.png",
        "Model Evaluation": "model_evaluation.png"
    }.items():
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(Image.open(path), width="stretch")
            st.markdown("---")
        else:
            st.warning(f"⚠️ {path} not found")

    st.markdown('<div class="section-label">Classifier Comparison</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Model':     ['Logistic Regression','Random Forest','Gradient Boosting','SVM','KNN','Decision Tree'],
        'Accuracy':  ['83%','85%','84%','83%','81%','78%'],
        'AUC Score': ['90%','92%','91%','90%','88%','78%'],
        'CV Score':  ['82%','84%','83%','82%','80%','76%'],
        'Verdict':   ['✅ Good','🏆 Best Model','✅ Good','✅ Good','✅ Good','⚠️ Fair']
    }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# TAB 4 — DATASET
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  len(df))
    c2.metric("Features",       df.shape[1]-1)
    c3.metric("Disease Cases",  disease)
    c4.metric("Missing Values", "None ✅")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️  Download Dataset as CSV",
                       df.to_csv(index=False),
                       'heart_disease_dataset.csv', 'text/csv')

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    🫀 <strong>CardioPredict</strong> — Heart Disease Prediction System
    <br>
    Machine Learning Based Clinical Decision Support Application
</div>
""", unsafe_allow_html=True)