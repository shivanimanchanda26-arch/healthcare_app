import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hospital Operations Portal", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
if "alert_status" not in st.session_state:
    st.session_state.alert_status = "Normal Operations"

# --- TOP HEADER & SEARCH ---
col_head1, col_head2, col_head3 = st.columns([3, 2, 1])

with col_head1:
    st.title("🏥 St. Jude Hospital Operations")
    st.caption(f"Status: **{st.session_state.alert_status}** | Active Shift: Morning (07:00 - 15:30)")

with col_head2:
    patient_search = st.text_input("Patient Lookup", placeholder="Enter Patient MRN or Name...")

with col_head3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚨 Trigger Code Blue", type="primary", use_container_width=True):
        st.session_state.alert_status = "CODE BLUE ACTIVE - RAPID RESPONSE"
        st.rerun()

st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🏢 Department Controls")
selected_ward = st.sidebar.selectbox(
    "Select Ward / Unit", 
    ["All Hospital Wards", "Emergency Department (ED)", "Intensive Care Unit (ICU)", "Surgery & Recovery", "General Pediatrics", "Maternity Ward"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Quick Actions")
if st.sidebar.button("➕ Admit New Patient", use_container_width=True):
    st.sidebar.success("Admission form dispatched to triage desk.")

if st.sidebar.button("📋 Generate Shift Report", use_container_width=True):
    st.sidebar.success("Shift handover summary compiled.")

if st.sidebar.button("🔄 Reset Alerts", use_container_width=True):
    st.session_state.alert_status = "Normal Operations"
    st.rerun()

# --- MOCK DATA GENERATION ---
@st.cache_data
def load_hospital_data():
    np.random.seed(404)
    n = 250
    return pd.DataFrame({
        "Patient_MRN": [f"MRN-{np.random.randint(10000, 99999)}" for _ in range(n)],
        "Department": np.random.choice(["Emergency", "ICU", "Surgery", "Pediatrics", "Maternity"], n, p=[0.35, 0.15, 0.2, 0.15, 0.15]),
        "Triage_Level": np.random.choice(["Level 1 (Critical)", "Level 2 (Emergent)", "Level 3 (Urgent)", "Level 4 (Standard)"], n, p=[0.1, 0.25, 0.4, 0.25]),
        "Wait_Time_Mins": np.random.exponential(scale=20, size=n).astype(int),
        "Bed_Assigned": np.random.choice([True, False], n, p=[0.8, 0.2]),
        "Satisfaction": np.random.randint(3, 6, n)
    })

df = load_hospital_data()

if selected_ward != "All Hospital Wards":
    dept_key = selected_ward.split(" ")[0]
    df = df[df['Department'].str.contains(dept_key, case=False, na=False)]

if patient_search:
    df = df[df['Patient_MRN'].str.contains(patient_search, case=False, na=False)]

# --- KEY PERFORMANCE METRICS (Native Streamlit) ---
m1, m2, m3, m4 = st.columns(4)

m1.metric(label="Bed Occupancy Rate", value="88.4%", delta="+2.1% from yesterday")
m2.metric(label="Average ED Wait Time", value="18 mins", delta="-4 mins", delta_color="inverse")
m3.metric(label="Active Admissions Today", value="142", delta="38 pending discharge")
m4.metric(label="Patient Satisfaction", value="4.7 / 5.0", delta="+0.3")

st.markdown("---")

# --- MAIN DASHBOARD CONTENT (Two Columns) ---
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📊 Departmental Bed Capacity")
    
    st.text("Emergency Department (92% Capacity)")
    st.progress(0.92)
    
    st.text("Intensive Care Unit - ICU (85% Capacity)")
    st.progress(0.85)
    
    st.text("Surgery & Recovery Wards (78% Capacity)")
    st.progress(0.78)
    
    st.text("General Pediatrics (64% Capacity)")
    st.progress(0.64)

with col_right:
    st.subheader("⚡ Hourly Patient Influx")
    hours = [f"{i}:00" for i in range(8, 20)]
    influx_vals = [12, 18, 30, 45, 52, 48, 40, 35, 28, 22, 16, 10]
    fig_line = px.line(x=hours, y=influx_vals, markers=True, labels={"x": "Hour", "y": "Patients Admitted"})
    fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# --- LOWER ANALYTICS ROW ---
g1, g2, g3 = st.columns(3)

with g1:
    st.subheader("🛡️ Triage Breakdown")
    triage_counts = df.groupby("Triage_Level")["Patient_MRN"].count().reset_index()
    fig_pie = px.pie(triage_counts, names="Triage_Level", values="Patient_MRN", hole=0.5)
    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
    st.plotly_chart(fig_pie, use_container_width=True)

with g2:
    st.subheader("🎯 Patient Record Queue")
    st.dataframe(df[["Patient_MRN", "Department", "Triage_Level", "Wait_Time_Mins"]].head(6), use_container_width=True, hide_index=True)

with g3:
    st.subheader("🧪 Wait Time Distribution")
    fig_box = px.box(df, x="Department", y="Wait_Time_Mins", color="Department")
    fig_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
