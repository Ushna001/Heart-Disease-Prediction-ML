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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding: 0rem 2.5rem 2rem 2.5rem; max-width: 1300px; }

    /* ─── GLOBAL ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* ─── HERO ─── */
    .hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 44px 52px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 100%;
        background: linear-gradient(160deg, #EFF6FF 0%, #E0E7FF 100%);
        clip-path: ellipse(100% 100% at 100% 50%);
    }
    .hero-eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 40px;
        font-weight: 400;
        line-height: 1.2;
        color: #0F172A;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
    }
    .hero-title span { color: #2563EB; }
    .hero-desc {
        font-size: 15px;
        color: #475569;
        line-height: 1.75;
        max-width: 540px;
        position: relative;
        z-index: 1;
        margin-bottom: 28px;
    }
    .hero-stats { display: flex; gap: 40px; position: relative; z-index: 1; }
    .hero-stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        color: #1D4ED8;
        line-height: 1;
        font-weight: 400;
    }
    .hero-stat-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ─── STAT CARDS ─── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
    }
    .stat-card.blue::after  { background: #2563EB; }
    .stat-card.red::after   { background: #DC2626; }
    .stat-card.green::after { background: #059669; }
    .stat-card.purple::after{ background: #7C3AED; }
    .stat-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-bottom: 12px;
    }
    .dot-blue   { background: #2563EB; }
    .dot-red    { background: #DC2626; }
    .dot-green  { background: #059669; }
    .dot-purple { background: #7C3AED; }
    .stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 30px;
        font-weight: 400;
        line-height: 1;
        color: #0F172A;
    }
    .stat-label {
        font-size: 11px;
        color: #64748B;
        margin-top: 5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 9px 20px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #64748B !important;
        background: transparent !important;
        letter-spacing: 0.1px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 6px rgba(29,78,216,0.25) !important;
    }

    /* ─── FORM SECTION HEADERS ─── */
    .form-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .form-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .form-dot-blue   { background: #2563EB; }
    .form-dot-teal   { background: #0D9488; }
    .form-dot-amber  { background: #D97706; }
    .form-section-title {
        font-size: 13px;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.1px;
    }
    .form-section-sub {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 1px;
    }

    /* ─── INPUT HINT ─── */
    .hint-bar {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 11px 15px;
        font-size: 13px;
        font-weight: 500;
        color: #1D4ED8;
        margin-bottom: 18px;
    }

    /* ─── BUTTONS ─── */
    .stButton > button {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        letter-spacing: 0.2px !important;
        transition: background 0.15s !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: #1E40AF !important;
    }
    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #1D4ED8 !important;
        border: 1.5px solid #1D4ED8 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        letter-spacing: 0.1px !important;
        transition: background 0.15s !important;
    }
    .stDownloadButton > button:hover {
        background: #EFF6FF !important;
    }

    /* ─── RESULT CARDS ─── */
    .result-wrap {
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        border: 1.5px solid;
    }
    .result-disease { background: #FFF5F5; border-color: #FECACA; }
    .result-healthy { background: #F0FDF4; border-color: #BBF7D0; }
    .result-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .badge-danger  { background: #FEE2E2; color: #B91C1C; }
    .badge-success { background: #DCFCE7; color: #15803D; }
    .result-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 24px;
        font-weight: 400;
        margin-bottom: 6px;
    }
    .result-disease .result-heading { color: #B91C1C; }
    .result-healthy .result-heading { color: #15803D; }
    .result-desc {
        font-size: 13.5px;
        color: #475569;
        line-height: 1.65;
    }
    .result-desc strong { color: #0F172A; }

    /* ─── PATIENT SUMMARY ─── */
    .summary-grid-modern {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 12px;
    }
    .summary-box {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        flex: 0 0 calc(50% - 6px);
        box-sizing: border-box;
    }
    .summary-label {
        font-size: 10px;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
    }
    .summary-data {
        font-size: 15px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 4px;
    }
    .risk-section {
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid #E2E8F0;
    }
    .risk-section h4 {
        margin-bottom: 12px;
        color: #1E293B;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .risk-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 16px;
    }
    .risk-metrics .summary-box {
        border-left-color: #7C3AED;
        flex: 0 0 calc(50% - 6px);
    }
    .recommendation-box {
        background: #F0FDF4;
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid #BBF7D0;
        border-left: 3px solid #059669;
    }
    .recommendation-title {
        font-weight: 700;
        font-size: 12px;
        color: #064E3B;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .recommendation-text {
        font-size: 13.5px;
        color: #1E293B;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ─── SECTION DIVIDER LABEL ─── */
    .section-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #CBD5E1;
        margin: 24px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #F1F5F9;
    }

    /* ─── SIDEBAR ─── */
    [data-testid="stSidebar"] {
        background: #0F172A !important;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    .sidebar-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 22px;
        color: #F8FAFC !important;
        text-align: center;
        padding: 20px 0 6px 0;
        letter-spacing: -0.3px;
    }
    .sidebar-tagline {
        font-size: 11px;
        color: #475569 !important;
        text-align: center;
        margin-bottom: 24px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sidebar-divider {
        height: 1px;
        background: #1E293B;
        margin: 14px 0;
    }
    .sidebar-section-title {
        font-size: 9px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: #334155 !important;
        margin-bottom: 8px !important;
    }
    .sidebar-item {
        font-size: 12.5px !important;
        color: #64748B !important;
        padding: 4px 0 !important;
    }

    /* ─── FOOTER ─── */
    .app-footer {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 28px;
        text-align: center;
        margin-top: 40px;
        color: #94A3B8;
        font-size: 12px;
        font-weight: 500;
    }
    .app-footer strong { color: #334155; }

    /* ─── STREAMLIT CONTROLS ─── */
    .stSlider label, .stSelectbox label {
        font-size: 12.5px !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    div[data-baseweb="select"] { border-radius: 7px !important; }
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
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

def generate_pdf(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, exercise_angina, oldpeak, st_segment, major_vessels, thal, probability, prediction, recommendation):
    from fpdf import FPDF
    
    class ClinicalSummaryPDF(FPDF):
        def header(self):
            # Banner background
            self.set_fill_color(26, 86, 219) # #1A56DB (Primary Blue)
            self.rect(0, 0, 210, 40, 'F')
            
            # Title
            self.set_text_color(255, 255, 255)
            self.set_font('helvetica', 'B', 20)
            self.set_y(10)
            self.cell(0, 10, 'CardioPredict Clinical Report', align='C', new_x="LMARGIN", new_y="NEXT")
            
            # Subtitle
            self.set_font('helvetica', 'I', 10)
            self.cell(0, 5, 'Cardiovascular Risk Assessment & Patient Summary Sheet', align='C', new_x="LMARGIN", new_y="NEXT")
            
            # Reset text color and positioning
            self.set_text_color(0, 0, 0)
            self.set_y(48)
            
        def footer(self):
            self.set_y(-20)
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, 'CardioPredict - Clinical Decision Support Application', align='C', new_x="LMARGIN", new_y="NEXT")
            self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C')

    pdf = ClinicalSummaryPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    # ── Section: Risk Assessment ──
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 8, '1. Risk Assessment Results', new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 86, 219)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    # Risk Box
    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
    bg_color = (254, 226, 226) if prediction == 1 else (220, 252, 231) # Light Red / Light Green
    text_color = (220, 38, 38) if prediction == 1 else (21, 128, 61) # Red / Green
    
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(*text_color)
    pdf.set_font('helvetica', 'B', 16)
    
    # Draw risk label container
    pdf.cell(0, 14, f'  Status: {risk_label}', border=1, fill=True, align='L', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Probability metrics
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(95, 8, f'Disease Probability: {probability[1]*100:.1f}%', border=1, align='C')
    pdf.cell(95, 8, f'Healthy Probability: {probability[0]*100:.1f}%', border=1, align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Clinical Recommendation
    pdf.set_fill_color(248, 250, 252) # #F8FAFC
    pdf.set_text_color(15, 23, 42) # #0F172A
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, '  Clinical Recommendation:', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('helvetica', '', 11)
    
    # Multi-line cell for recommendation text
    pdf.multi_cell(0, 7, f'  {recommendation}', border='L', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    
    # ── Section: Patient Demographics & Clinical Variables ──
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 8, '2. Patient Clinical Variables', new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 86, 219)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    # Data table for parameters
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 10)
    
    sex_str = "Male" if sex == 1 else "Female"
    cp_str = {1:'Typical Angina', 2:'Atypical Angina', 3:'Non-Anginal Pain', 4:'Asymptomatic'}[chest_pain]
    fbs_str = "Yes" if fasting_bs == 1 else "No"
    angina_str = "Yes" if exercise_angina == 1 else "No"
    ecg_str = {0:'Normal', 1:'ST-T Abnormality', 2:'LV Hypertrophy'}[rest_ecg]
    slope_str = {1:'Upsloping', 2:'Flat', 3:'Downsloping'}[st_segment]
    thal_str = {3:'Normal', 6:'Fixed Defect', 7:'Reversible Defect'}[thal]
    
    # We will build a clean 2-column table key-value pairs
    items = [
        ("Age", f"{age} Years", "Blood Pressure", f"{resting_bp} mmHg"),
        ("Sex", sex_str, "Serum Cholesterol", f"{cholesterol} mg/dL"),
        ("Chest Pain Type", cp_str, "Maximum Heart Rate", f"{max_hr} bpm"),
        ("Fasting Blood Sugar", fbs_str, "Exercise Angina", angina_str),
        ("Oldpeak (ST Depression)", f"{oldpeak:.2f}", "ST Segment Slope", slope_str),
        ("Major Vessels Colored", f"{major_vessels}", "Thalassemia Type", thal_str)
    ]
    
    pdf.set_fill_color(245, 247, 250) # light zebra lines
    fill = False
    
    # Table header
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(45, 8, ' Parameter', border=1, fill=True)
    pdf.cell(50, 8, ' Value', border=1, fill=True)
    pdf.cell(45, 8, ' Parameter', border=1, fill=True)
    pdf.cell(50, 8, ' Value', border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_fill_color(248, 250, 252)
    
    for item in items:
        pdf.cell(45, 8, f' {item[0]}', border=1, fill=fill)
        pdf.cell(50, 8, f' {item[1]}', border=1, fill=fill)
        pdf.cell(45, 8, f' {item[2]}', border=1, fill=fill)
        pdf.cell(50, 8, f' {item[3]}', border=1, fill=fill, new_x="LMARGIN", new_y="NEXT")
        fill = not fill
        
    pdf.ln(10)
    
    # Disclaimer block
    pdf.set_fill_color(255, 251, 235) # Light amber background for warnings
    pdf.set_draw_color(245, 158, 11) # Amber border
    pdf.set_text_color(180, 83, 9) # Amber text
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(0, 6, '  [!] CLINICAL DISCLAIMER & USE NOTICE:', border='TLR', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(0, 5, '  This report is generated by a machine learning classifier for clinical decision support purposes and should not be used as the sole basis for diagnosis. All risk scores must be clinically validated by a qualified healthcare professional in combination with patient medical history and additional laboratory diagnostics.', border='BLR', fill=True, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

model, scaler = load_model()
df = load_data()

total   = len(df)
disease = int(df['target'].sum())
healthy = int((df['target'] == 0).sum())

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">CardioPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Clinical Decision Support</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Dataset Info</div>', unsafe_allow_html=True)
    for item in [f"{total} patient records", "13 clinical features",
                 "No missing values", "UCI Heart Disease Dataset"]:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Author</div>', unsafe_allow_html=True)
    st.markdown("""<div style='font-size:12px; color:#475569; line-height:2;'>
    <b style='color:#CBD5E1'>M U F Ushna</b><br>
    IT23436312 &middot; SLIIT<br>
    <a href='https://github.com/Ushna001/Heart-Disease-Prediction-ML'
       style='color:#60A5FA; text-decoration:none; font-weight:600;'>GitHub Repository</a></div>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning &nbsp;&middot;&nbsp; Clinical AI &nbsp;&middot;&nbsp; Cardiology</div>
    <div class="hero-title">Cardiovascular Risk<br>Assessment with <span>Clinical Precision</span></div>
    <div class="hero-desc">
        Enter patient clinical parameters below to receive an instant risk assessment
        powered by a trained machine learning classifier validated on the UCI Heart Disease dataset.
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
            <div class="hero-stat-label">AUC-ROC Score</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS ───────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card blue">
        <span class="stat-dot dot-blue"></span>
        <div class="stat-num">{total}</div>
        <div class="stat-label">Total Patients</div>
    </div>
    <div class="stat-card red">
        <span class="stat-dot dot-red"></span>
        <div class="stat-num" style="color:#DC2626">{disease}</div>
        <div class="stat-label">Disease Cases</div>
    </div>
    <div class="stat-card green">
        <span class="stat-dot dot-green"></span>
        <div class="stat-num" style="color:#059669">{healthy}</div>
        <div class="stat-label">Healthy Cases</div>
    </div>
    <div class="stat-card purple">
        <span class="stat-dot dot-purple"></span>
        <div class="stat-num" style="color:#7C3AED">85%</div>
        <div class="stat-label">Best Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Data Analysis",
    "Model Performance",
    "Dataset"
])

# ═══════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Patient Clinical Input</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hint-bar">
        Fill in all clinical fields below. Values are pre-set to dataset averages as defaults.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="form-section-header">
            <span class="form-dot form-dot-blue"></span>
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
            <span class="form-dot form-dot-teal"></span>
            <div>
                <div class="form-section-title">Measurements</div>
                <div class="form-section-sub">Vitals &amp; lab results</div>
            </div>
        </div>""", unsafe_allow_html=True)
        resting_bp  = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 131)
        cholesterol = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 249)
        max_hr      = st.slider("Max Heart Rate Achieved (bpm)", 70, 210, 149)
        oldpeak     = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.05, step=0.1)

    with col3:
        st.markdown("""
        <div class="form-section-header">
            <span class="form-dot form-dot-amber"></span>
            <div>
                <div class="form-section-title">Diagnostics</div>
                <div class="form-section-sub">ECG &amp; imaging findings</div>
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

    if st.button("Run Cardiovascular Risk Assessment"):
        cols = ['age', 'sex', 'chest pain type', 'resting blood pressure',
                'serum cholestoral', 'fasting blood sugar',
                'resting electrocardiographic results', 'max heart rate',
                'exercise induced angina', 'oldpeak', 'ST segment',
                'major vessels', 'thal']
        inp = pd.DataFrame([[age, sex, chest_pain, resting_bp, cholesterol,
                             fasting_bs, rest_ecg, max_hr, exercise_angina,
                             oldpeak, st_segment, major_vessels, thal]],
                            columns=cols)
        inp_scaled  = scaler.transform(inp)
        prediction  = model.predict(inp_scaled)[0]
        probability = model.predict_proba(inp_scaled)[0]

        st.markdown('<div class="section-label">Risk Assessment Results</div>', unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-wrap result-disease">
                    <span class="result-badge badge-danger">High Risk</span>
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
                    <span class="result-badge badge-success">Low Risk</span>
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

        st.markdown('<div class="section-label">Patient Summary Report</div>', unsafe_allow_html=True)

        recommendation = (
            "Immediate cardiology consultation and additional diagnostic testing are recommended."
            if prediction == 1 else
            "Continue healthy lifestyle habits and schedule routine cardiovascular checkups."
        )

        st.markdown(f"""
        <div class="summary-grid-modern">
            <div class="summary-box">
                <div class="summary-label">Age</div>
                <div class="summary-data">{age} Years</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Sex</div>
                <div class="summary-data">
                    {"Male" if sex == 1 else "Female"}
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Chest Pain Type</div>
                <div class="summary-data">
                    { {1:'Typical Angina',2:'Atypical Angina',3:'Non-Anginal Pain',4:'Asymptomatic'}[chest_pain] }
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Blood Pressure</div>
                <div class="summary-data">{resting_bp} mmHg</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Cholesterol</div>
                <div class="summary-data">{cholesterol} mg/dL</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Maximum Heart Rate</div>
                <div class="summary-data">{max_hr} bpm</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Fasting Blood Sugar</div>
                <div class="summary-data">
                    {"Yes" if fasting_bs == 1 else "No"}
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Exercise Angina</div>
                <div class="summary-data">
                    {"Yes" if exercise_angina == 1 else "No"}
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Oldpeak</div>
                <div class="summary-data">{oldpeak}</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">ST Segment</div>
                <div class="summary-data">
                    { {1:'Upsloping',2:'Flat',3:'Downsloping'}[st_segment] }
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Major Vessels</div>
                <div class="summary-data">{major_vessels}</div>
            </div>
            <div class="summary-box">
                <div class="summary-label">Thalassemia</div>
                <div class="summary-data">
                    { {3:'Normal',6:'Fixed Defect',7:'Reversible Defect'}[thal] }
                </div>
            </div>
        </div>

        <div class="risk-section">
            <h4>Risk Assessment</h4>
            <div class="risk-metrics">
                <div class="summary-box">
                    <div class="summary-label">Disease Probability</div>
                    <div class="summary-data">{probability[1]*100:.1f}%</div>
                </div>
                <div class="summary-box">
                    <div class="summary-label">Healthy Probability</div>
                    <div class="summary-data">{probability[0]*100:.1f}%</div>
                </div>
            </div>
            <div class="recommendation-box">
                <div class="recommendation-title">Clinical Recommendation</div>
                <div class="recommendation-text">{recommendation}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pdf_bytes = generate_pdf(
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, rest_ecg, max_hr, exercise_angina,
            oldpeak, st_segment, major_vessels, thal,
            probability, prediction, recommendation
        )

        st.download_button(
            label="Download Clinical Summary Report as PDF",
            data=pdf_bytes,
            file_name=f"cardiopredict_report_{age}_{'M' if sex==1 else 'F'}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

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
            st.image(Image.open(path), use_container_width=True)
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
            st.image(Image.open(path), use_container_width=True)
            st.markdown("---")
        else:
            st.warning(f"⚠️ {path} not found")

    st.markdown('<div class="section-label">Classifier Comparison</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Model':     ['Logistic Regression','Random Forest','Gradient Boosting','SVM','KNN','Decision Tree'],
        'Accuracy':  ['83%','85%','84%','83%','81%','78%'],
        'AUC Score': ['90%','92%','91%','90%','88%','78%'],
        'CV Score':  ['82%','84%','83%','82%','80%','76%'],
        'Verdict':   ['Good','Best Model','Good','Good','Good','Fair']
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
    c4.metric("Missing Values", "None")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Dataset as CSV",
                       df.to_csv(index=False),
                       'heart_disease_dataset.csv', 'text/csv')

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong>CardioPredict</strong> &mdash; Heart Disease Prediction System<br>
    Machine Learning Based Clinical Decision Support &middot; UCI Heart Disease Dataset
</div>
""", unsafe_allow_html=True)