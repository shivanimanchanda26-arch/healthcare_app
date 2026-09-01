import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hospital Operations & Patient Care Portal", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CLINICAL ADMIN THEME CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b132b !important;
        color: #e0fbfc !important;
    }
    .main {
        background-color: #0b132b !important;
        color: #e0fbfc !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1c2541 !important;
        border-right: 1px solid #3a506b !important;
    }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div, [data-testid="stSidebar"] p {
        color: #e0fbfc !important;
    }
    .card {
        background-color: #1c2541 !important;
        border: 1px solid #3a506b !important;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    p, span, label, div {
        color: #e0fbfc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# --- TOP NAVIGATION & SEARCH BAR ---
top_col1, top_col2, top_col3, top_col4 = st.columns([3, 2, 2, 1])

with top_col1:
    st.markdown("### 🏥 ST. JUDE HOSPITAL OPERATIONS & CARE PORTAL")

with top_col2:
    patient_search = st.text_input("Patient Lookup", placeholder="Search patient name or MRN...", label_visibility="collapsed")

with top_col3:
    if st.button("📋 Export Shift Handover Report", use_container_width=True):
        st.session_state.report_generated = True
        st.success("Shift report compiled for nursing staff.")

with top_col4:
    st.markdown("<div style='background:#48cae4; color:#0b132b; padding:8px 12px; border-radius:50%; font-weight:bold; text-align:center; width:36px;'>ADM</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #3a506b;'>", unsafe_allow_html=True)

# --- LEFT SIDEBAR: HOSPITAL FILTERS & CONTROLS ---
st.sidebar.markdown("### 🏢 WARD & DEPARTMENT")
selected_ward = st.sidebar.selectbox("Active Department", ["All Hospital Wards", "Emergency Department (ED)", "Intensive Care Unit (ICU)", "Surgery & Recovery", "General Pediatrics", "Maternity Ward"])

st.sidebar.markdown("<br><hr style='border-color: #3a506b;'>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚡ QUICK ACTIONS")
if st.sidebar.button("➕ Admit New Patient", use_container_width=True):
    st.sidebar.success("Admission triage form opened.")

if st.sidebar.button("🚨 Request Emergency Transfer", use_container_width=True):
    st.sidebar.warning("Rapid Response Team alerted.")

shift_timing = st.sidebar.selectbox("Current Shift", ["Morning Shift (07:00 - 15:30)", "Evening Shift (15:00 - 23:30)", "Night Shift (23:00 - 07:30)"])

# --- GENERATE MOCK HOSPITAL OPERATIONAL DATA ---
@st.cache_data
def load_hospital_data():
    np.random.seed(303)
    n = 300
    return pd.DataFrame({
        "Patient_MRN": [f"MRN-{np.random.randint(10000, 99999)}" for _ in range(n)],
        "Department": np.random.choice(["Emergency", "ICU", "Surgery", "Pediatrics", "Maternity"], n, p=[0.35, 0.15, 0.2, 0.15, 0.15]),
        "Triage_Priority": np.random.choice(["Level 1 (Resuscitation)", "Level 2 (Emergent)", "Level 3 (Urgent)", "Level 4 (Standard)"], n, p=[0.1, 0.25, 0.4, 0.25]),
        "Wait_Time_Mins": np.random.exponential(scale=25, size=n).astype(int),
        "Bed_Assigned": np.random.choice([True, False], n, p=[0.75, 0.25]),
        "Satisfaction_Score": np.random.randint(3, 6, n)
    })

df_hospital = load_hospital_data()

if selected_ward != "All Hospital Wards":
    dept_short = selected_ward.split(" ")[0]
    df_hospital = df_hospital[df_hospital['Department'].str.contains(dept_short, case=False, na=False)]

if patient_search:
    df_hospital = df_hospital[df_hospital['Patient_MRN'].str.contains(patient_search, case=False, na=False)]

# --- TOP KPI METRICS ROW ---
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #48cae4;">
        <div style="color: #90e0ef; font-size: 11px; text-transform: uppercase; font-weight: bold;">Bed Occupancy Rate</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">88.4%</div>
        <div style="color: #48cae4; font-size: 11px; margin-top: 5px;">42 BEDS AVAILABLE SYSTEM-WIDE</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #00b4d8;">
        <div style="color: #90e0ef; font-size: 11px; text-transform: uppercase; font-weight: bold;">Average ED Wait Time</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">18 mins</div>
        <div style="color: #00b4d8; font-size: 11px; margin-top: 5px;">-4 MINS VS PREVIOUS HOUR</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #90e0ef;">
        <div style="color: #90e0ef; font-size: 11px; text-transform: uppercase; font-weight: bold;">Active Admissions Today</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">142</div>
        <div style="color: #90e0ef; font-size: 11px; margin-top: 5px;">38 DISCHARGES PENDING</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="card" style="border-left: 4px solid #ade8f4;">
        <div style="color: #90e0ef; font-size: 11px; text-transform: uppercase; font-weight: bold;">Patient Satisfaction Score</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">4.7 / 5.0</div>
        <div style="color: #ade8f4; font-size: 11px; margin-top: 5px;">BASED ON 310 EXIT SURVEYS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- SECONDARY DASHBOARD ROW ---
col_c1, col_c2 = st.columns([1.3, 1])

with col_c1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 12px;">📊 Departmental Capacity & Bed Utilization</div>
        <span style='color: #e0fbfc; font-size: 12px;'>Emergency Department (92% Capacity)</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(0.92)
    
    st.markdown("<span style='color: #e0fbfc; font-size: 12px;'>Intensive Care Unit - ICU (85% Capacity)</span>", unsafe_allow_html=True)
    st.progress(0.85)
    
    st.markdown("<span style='color: #e0fbfc; font-size: 12px;'>Surgery & Recovery Wards (78% Capacity)</span>", unsafe_allow_html=True)
    st.progress(0.78)

with col_c2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">⚡ Hourly Patient Influx Stream</div>
        <div style="font-size: 20px; font-weight: bold; color: #48cae4;">Peak Admission Hour: 14:00</div>
    </div>
    """, unsafe_allow_html=True)
    x_influx = np.arange(12)
    y_influx = np.array([12, 15, 22, 35, 48, 42, 38, 30, 25, 20, 18, 14]) + np.random.normal(0, 2, 12)
    fig_influx = go.Figure(go.Scatter(x=x_influx, y=y_influx, mode='lines+markers', line=dict(color='#48cae4', width=2.5), fill='tozeroy', fillcolor='rgba(72, 202, 228, 0.2)'))
    fig_influx.update_layout(paper_bgcolor='#1c2541', plot_bgcolor='#1c2541', margin=dict(t=5, b=5, l=5, r=5), height=115, font=dict(color="#e0fbfc"), xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_influx, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LOWER ANALYTICS ROW ---
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🛡️ Patient Triage Priority Distribution</div>
    </div>
    """, unsafe_allow_html=True)
    triage_counts = df_hospital.groupby("Triage_Priority")["Patient_MRN"].count().reset_index()
    fig_donut = px.pie(triage_counts, names="Triage_Priority", values="Patient_MRN", hole=0.6, color_discrete_sequence=["#ff758f", "#ffb703", "#90e0ef", "#48cae4"])
    fig_donut.update_layout(paper_bgcolor='#1c2541', plot_bgcolor='#1c2541', margin=dict(t=10, b=10, l=10, r=10), height=180, font=dict(color="#e0fbfc"), legend=dict(orientation="h", y=-0.25, font=dict(color="#e0fbfc", size=9)))
    st.plotly_chart(fig_donut, use_container_width=True)

with g2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🎯 Overall Ward Occupancy Gauge</div>
    </div>
    """, unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=88.4,
        number={'suffix': "%", 'font': {'size': 20, 'color': '#ffffff'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#48cae4'}, 'bgcolor': "#3a506b", 'borderwidth': 0, 'steps': [{'range': [0, 100], 'color': '#3a506b'}]}
    ))
    fig_gauge.update_layout(paper_bgcolor='#1c2541', plot_bgcolor='#1c2541', margin=dict(t=25, b=10, l=10, r=10), height=155, font=dict(color="#e0fbfc"))
    st.plotly_chart(fig_gauge, use_container_width=True)

with g3:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🧪 Wait Time Spread Across Wards</div>
    </div>
    """, unsafe_allow_html=True)
    fig_box = px.box(df_hospital, x="Department", y="Wait_Time_Mins", color="Department", color_discrete_sequence=["#48cae4", "#00b4d8", "#90e0ef", "#ade8f4", "#ffb703"])
    fig_box.update_layout(paper_bgcolor='#1c2541', plot_bgcolor='#1c2541', margin=dict(t=5, b=5, l=5, r=5), height=170, font=dict(color="#e0fbfc"), showlegend=False, xaxis=dict(title=None), yaxis=dict(title=None))
    st.plotly_chart(fig_box, use_container_width=True)
