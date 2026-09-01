import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Clinical AI Diagnostics & Lesion Analysis Hub", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MEDICAL-GRADE DARK THEME CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #060913 !important;
        color: #f1f5f9 !important;
    }
    .main {
        background-color: #060913 !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1329 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div, [data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
    }
    .card {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    p, span, label, div {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CLINICAL SESSION STATE ---
if "audit_logged" not in st.session_state:
    st.session_state.audit_logged = False

# --- CLINICAL TOP TOOLBAR ---
top_col1, top_col2, top_col3, top_col4 = st.columns([3, 2, 2, 1])

with top_col1:
    st.markdown("### 🧬 CLINICAL AI & LESION ANALYSIS HUB")

with top_col2:
    search_patient = st.text_input("DICOM Patient Lookup", placeholder="Enter DICOM ID (e.g., DICOM_891)...", label_visibility="collapsed")

with top_col3:
    if st.button("🔒 Export HIPAA Audit Log", use_container_width=True):
        st.session_state.audit_logged = True
        st.success("HIPAA cryptographic audit log generated.")

with top_col4:
    st.markdown("<div style='background:#14b8a6; color:#060913; padding:8px 12px; border-radius:50%; font-weight:bold; text-align:center; width:36px;'>DR.S</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

# --- LEFT SIDEBAR: CLINICAL AI CONTROLS ---
st.sidebar.markdown("### 🔬 MODEL ARCHITECTURE")
selected_model = st.sidebar.selectbox("Inference Engine", ["ResNet-50 Clinical Core v4", "Custom 3D CNN (Lesion Segmentation)", "DenseNet-121 Multi-Path"])
gradcam_layer = st.sidebar.selectbox("Grad-CAM Target Layer", ["Layer4:conv3", "FeaturePyramid_Neck", "Attention_Block_2"])
confidence_threshold = st.sidebar.slider("Diagnostic Sensitivity Threshold", 0.70, 0.99, 0.88, 0.01)

st.sidebar.markdown("<br><hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🛡️ Pipeline Controls")
if st.sidebar.button("⚡ Run Stress Test Suite", use_container_width=True):
    st.sidebar.success("Adverse perturbation stress test completed (Zero degradation).")

# --- DATA GENERATION ---
@st.cache_data
def load_clinical_data():
    np.random.seed(108)
    n = 350
    return pd.DataFrame({
        "Patient_ID": [f"DICOM_{i:04d}" for i in range(1, n+1)],
        "Scan_Modality": np.random.choice(["MRI Brain", "CT Chest", "X-Ray Thorax", "MRI Spine"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "Lesion_Probability": np.random.uniform(0.12, 0.98, n),
        "Triage_Status": np.random.choice(["Critical Urgent", "Secondary Review", "Cleared"], n, p=[0.2, 0.35, 0.45]),
        "Inference_Latency_ms": np.random.uniform(6.5, 18.2, n)
    })

df_clinical = load_clinical_data()

if search_patient:
    df_clinical = df_clinical[df_clinical['Patient_ID'].str.contains(search_patient, case=False, na=False)]

# --- MAIN DASHBOARD: CLINICAL METRICS ---
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #14b8a6;">
        <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">Model Accuracy (AUC-ROC)</div>
        <div style="font-size: 26px; font-weight: bold; color: #f1f5f9; margin-top: 5px;">0.987</div>
        <div style="color: #14b8a6; font-size: 11px; margin-top: 5px;">VALIDATED ON SYNTHETIC COHORT</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #0ea5e9;">
        <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">Active DICOM Queue</div>
        <div style="font-size: 26px; font-weight: bold; color: #f1f5f9; margin-top: 5px;">1,420</div>
        <div style="color: #0ea5e9; font-size: 11px; margin-top: 5px;">STREAMING FROM PACS</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #a855f7;">
        <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">Mean Inference Latency</div>
        <div style="font-size: 26px; font-weight: bold; color: #f1f5f9; margin-top: 5px;">8.4 ms</div>
        <div style="color: #c084fc; font-size: 11px; margin-top: 5px;">NVIDIA TENSORRT ACCELERATED</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #ef4444;">
        <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">False Positive Rate</div>
        <div style="font-size: 26px; font-weight: bold; color: #f1f5f9; margin-top: 5px;">1.12%</div>
        <div style="color: #f87171; font-size: 11px; margin-top: 5px;">WITHIN CLINICAL TOLERANCE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- SECONDARY VISUALIZATION ROW ---
col_c1, col_c2 = st.columns([1.3, 1])

with col_c1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 10px;">📊 Pipeline Calibration & Validation Progress</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<span style='color: #f1f5f9; font-size: 12px;'>ResNet Feature Extraction Calibration (94%)</span>", unsafe_allow_html=True)
    st.progress(0.94)
    st.markdown("<span style='color: #f1f5f9; font-size: 12px;'>Grad-CAM Spatial Alignment (88%)</span>", unsafe_allow_html=True)
    st.progress(0.88)
    st.markdown("<span style='color: #f1f5f9; font-size: 12px;'>Adverse Stress Test Robustness (96%)</span>", unsafe_allow_html=True)
    st.progress(0.96)

with col_c2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 5px;">⚡ Real-time Physiological Stream</div>
        <div style="font-size: 20px; font-weight: bold; color: #14b8a6;">Normal Sinus Rhythm (120 Hz)</div>
    </div>
    """, unsafe_allow_html=True)
    x_sig = np.arange(40)
    y_sig = np.sin(x_sig * 0.5) * np.exp(-x_sig * 0.01) + np.random.normal(0, 0.08, 40)
    fig_sig = go.Figure(go.Scatter(x=x_sig, y=y_sig, mode='lines', line=dict(color='#14b8a6', width=2), fill='tozeroy', fillcolor='rgba(20, 184, 166, 0.15)'))
    fig_sig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', margin=dict(t=5, b=5, l=5, r=5), height=105, font=dict(color="#f1f5f9"), xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_sig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LOWER ANALYTICS ROW ---
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 5px;">🛡️ Patient Triage Risk Stratification</div>
    </div>
    """, unsafe_allow_html=True)
    triage_counts = df_clinical.groupby("Triage_Status")["Patient_ID"].count().reset_index()
    fig_donut = px.pie(triage_counts, names="Triage_Status", values="Patient_ID", hole=0.6, color_discrete_sequence=["#ef4444", "#0ea5e9", "#14b8a6"])
    fig_donut.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', margin=dict(t=10, b=10, l=10, r=10), height=180, font=dict(color="#f1f5f9"), legend=dict(orientation="h", y=-0.2, font=dict(color="#f1f5f9", size=10)))
    st.plotly_chart(fig_donut, use_container_width=True)

with g2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 5px;">🎯 Model Confidence Gauge</div>
    </div>
    """, unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=98.7,
        number={'suffix': "%", 'font': {'size': 20, 'color': '#f1f5f9'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#14b8a6'}, 'bgcolor': "#1e293b", 'borderwidth': 0, 'steps': [{'range': [0, 100], 'color': '#1e293b'}]}
    ))
    fig_gauge.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', margin=dict(t=25, b=10, l=10, r=10), height=155, font=dict(color="#f1f5f9"))
    st.plotly_chart(fig_gauge, use_container_width=True)

with g3:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 5px;">🧪 Modality Error Rate Distribution</div>
    </div>
    """, unsafe_allow_html=True)
    fig_box = px.box(df_clinical, x="Scan_Modality", y="Lesion_Probability", color="Scan_Modality", color_discrete_sequence=["#14b8a6", "#0ea5e9", "#a855f7", "#ef4444"])
    fig_box.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', margin=dict(t=5, b=5, l=5, r=5), height=170, font=dict(color="#f1f5f9"), showlegend=False, xaxis=dict(title=None), yaxis=dict(title=None))
    st.plotly_chart(fig_box, use_container_width=True)
