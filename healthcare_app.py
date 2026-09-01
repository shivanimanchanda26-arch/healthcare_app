import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Trustworthy Medical Image Diagnosis Engine", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #090b10 0%, #131722 50%, #0d1017 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f3f4f6;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1017;
        border-right: 1px solid #1f293d;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #d1d5db !important;
    }
    .top-navbar {
        background-color: #131824;
        padding: 12px 20px;
        border-radius: 12px;
        border: 1px solid #1f293d;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .card {
        background: linear-gradient(145deg, #161c2d, #101420);
        border: 1px solid #1f293d;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 16px;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600;
    }
    p, span, label {
        color: #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-navbar">
    <div style="font-weight: bold; color: #ffffff; font-size: 18px; letter-spacing: 0.5px;">🏥 MEDICAL DIAGNOSIS DASHBOARD</div>
    <div style="color: #9ca3af; font-size: 13px; background: #090b10; padding: 6px 16px; border-radius: 8px; border: 1px solid #1f293d; width: 260px; text-align: center;">🔍 Search patient records...</div>
    <div style="display: flex; align-items: center; gap: 15px; color: #d1d5db; font-size: 14px;">
        <span>🔍</span><span>✉️</span><span>🔔</span><span>⚙️</span>
        <b style="color: #ffffff;">DR. SHIVANI M.</b>
        <div style="width: 32px; height: 32px; background: #38bdf8; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #090b10; font-weight: bold;">SM</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚡ NAVIGATION")
nav_selection = st.sidebar.radio(
    "Menu", 
    ["🏠 Main Dashboard", "📊 Scan Analytics", "📈 Reliability Metrics", "⚙️ System Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br><hr style='border-color: #1f293d;'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🛡️ Analysis Controls")
selected_arch = st.sidebar.selectbox("AI Model Version", ["Advanced ResNet Structure", "Standard Baseline Model"])
calibration_temp = st.sidebar.slider("Confidence Tuning Level", 0.5, 2.5, 1.2, 0.1)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        "ID": [f"PAT_{i:03d}" for i in range(1, n+1)],
        "Metric_Type": np.random.choice(["Error Rate", "Clarity Check", "Stress Test"], n),
        "Value": np.random.uniform(0.75, 0.99, n),
        "Category": np.random.choice(["High Risk", "Moderate Risk", "Stable"], n, p=[0.25, 0.35, 0.4])
    })

df = load_data()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card" style="border-top: 4px solid #facc15;">
        <div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Diagnostic Accuracy</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">98.5%</div>
        <div style="color: #facc15; font-size: 11px; margin-top: 5px;">VERIFIED MODEL SCORE</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card" style="border-top: 4px solid #38bdf8;">
        <div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Active Cases</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">2,481</div>
        <div style="color: #38bdf8; font-size: 11px; margin-top: 5px;">CURRENT QUEUE</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card" style="border-top: 4px solid #a855f7;">
        <div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Scans Processed Today</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">31,124</div>
        <div style="color: #c084fc; font-size: 11px; margin-top: 5px;">GPU ACCELERATED</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card" style="border-top: 4px solid #ec4899;">
        <div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Processing Speed</div>
        <div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">14.2 ms</div>
        <div style="color: #f472b6; font-size: 11px; margin-top: 5px;">AVERAGE RESPONSE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1.2, 1])

with col_m1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 12px;">📈 Scan Quality & Verification Progress</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<span style='color: #d1d5db; font-size: 12px;'>Batch Validation Score (82%)</span>", unsafe_allow_html=True)
    st.progress(0.82)
    st.markdown("<span style='color: #d1d5db; font-size: 12px;'>Confidence Alignment (65%)</span>", unsafe_allow_html=True)
    st.progress(0.65)
    st.markdown("<span style='color: #d1d5db; font-size: 12px;'>Reliability Index (91%)</span>", unsafe_allow_html=True)
    st.progress(0.91)

with col_m2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">⚡ Live System Signal Quality</div>
        <div style="font-size: 22px; font-weight: bold; color: #ffffff;">476 Hz</div>
    </div>
    """, unsafe_allow_html=True)
    x_w = np.arange(30)
    y_w = np.sin(x_w * 0.4) * np.exp(-x_w * 0.02) + np.random.normal(0, 0.1, 30)
    fig_w = go.Figure(go.Scatter(x=x_w, y=y_w, mode='lines', line=dict(color='#a855f7', width=2), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.2)'))
    fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=5, b=5, l=5, r=5), height=110, font_color="#f3f4f6", xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_w, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🛡️ Patient Risk Level Breakdown</div>
    </div>
    """, unsafe_allow_html=True)
    counts = df.groupby("Category")["ID"].count().reset_index()
    fig_donut = px.pie(counts, names="Category", values="ID", hole=0.65, color_discrete_sequence=["#38bdf8", "#ec4899", "#facc15"])
    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), height=180, font_color="#f3f4f6", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_donut, use_container_width=True)

with g2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🎯 Prediction Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=180347,
        number={'prefix': "$", 'font': {'size': 18, 'color': '#ffffff'}},
        gauge={'axis': {'range': [0, 200000]}, 'bar': {'color': '#38bdf8'}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0, 'steps': [{'range': [0, 200000], 'color': '#1f293d'}]}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=10, l=10, r=10), height=160, font_color="#f3f4f6")
    st.plotly_chart(fig_gauge, use_container_width=True)

with g3:
    st.markdown("""
    <div class="card">
        <div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">🧪 Image Quality Stress Test</div>
    </div>
    """, unsafe_allow_html=True)
    x_t = np.arange(40)
    y_t = np.random.uniform(0.2, 0.9, 40)
    fig_bar = px.bar(x=x_t, y=y_t, color_discrete_sequence=["#38bdf8"])
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=5, b=5, l=5, r=5), height=170, font_color="#f3f4f6", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
