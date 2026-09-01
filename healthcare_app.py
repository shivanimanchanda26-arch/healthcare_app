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
        background: linear-gradient(135deg, #0d0f18 0%, #171b2e 50%, #121522 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background-color: #111420;
        border-right: 1px solid #1f2437;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #9ca3af !important;
    }
    .top-navbar {
        background-color: #161a29;
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid #1f2437;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .card {
        background: linear-gradient(145deg, #181d2e, #131724);
        border: 1px solid #1f2437;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 15px;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-navbar">
    <div style="font-weight: bold; color: #ffffff; font-size: 20px; letter-spacing: 0.5px;">DASHBOARD</div>
    <div style="color: #9ca3af; font-size: 14px; background: #0f111a; padding: 6px 15px; border-radius: 8px; border: 1px solid #1f2437; width: 280px; text-align: center;">🔍 Search clinical logs...</div>
    <div style="display: flex; align-items: center; gap: 15px; color: #9ca3af; font-size: 14px;">
        <span>🔍</span><span>✉️</span><span>🔔</span><span>⚙️</span>
        <b style="color: #ffffff;">JOHN SMITH</b>
        <div style="width: 32px; height: 32px; background: #38bdf8; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #0d0f18; font-weight: bold;">JS</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚡ MENU")
nav_selection = st.sidebar.radio("Navigation", ["🏠 Dashboard", "📊 Charts", "📈 Statistic", "📅 Calendar", "📄 Sample Pages", "💬 Messages", "⚙️ Settings", "❓ Support"], label_visibility="collapsed")
st.sidebar.markdown("<br><hr style='border-color: #1f2437;'>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Governance Controls")
selected_arch = st.sidebar.selectbox("Model Architecture", ["ResNet Bottleneck Structure", "Vanilla CNN Baseline"])
calibration_temp = st.sidebar.slider("Temperature Scaling (T)", 0.5, 2.5, 1.2, 0.1)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        "ID": [f"PAT_{i:03d}" for i in range(1, n+1)],
        "Metric_Type": np.random.choice(["ECE Loss", "Grad-CAM", "Noise Stress"], n),
        "Value": np.random.uniform(0.75, 0.99, n),
        "Category": np.random.choice(["High Risk", "Moderate", "Stable"], n, p=[0.25, 0.35, 0.4])
    })

df = load_data()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="card" style="border-top: 4px solid #facc15;"><div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Model Accuracy</div><div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">98,5%</div><div style="color: #facc15; font-size: 11px; margin-top: 5px;">LOREM IPSUM DOLOR</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card" style="border-top: 4px solid #38bdf8;"><div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Active Cases</div><div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">2 481</div><div style="color: #38bdf8; font-size: 11px; margin-top: 5px;">ADIPISCING ELIT</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card" style="border-top: 4px solid #a855f7;"><div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Daily Inferences</div><div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">31 124</div><div style="color: #c084fc; font-size: 11px; margin-top: 5px;">DEXO EIUSMOD</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="card" style="border-top: 4px solid #ec4899;"><div style="color: #9ca3af; font-size: 11px; text-transform: uppercase; font-weight: bold;">Cost Optimization</div><div style="font-size: 26px; font-weight: bold; color: #ffffff; margin-top: 5px;">$ 2 125</div><div style="color: #f472b6; font-size: 11px; margin-top: 5px;">TEMPOR INCIDIDUNT</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1.2, 1])
with col_m1:
    st.markdown("""<div class="card"><div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 10px;">DOLOR — Calibration Progress Lines</div></div>""", unsafe_allow_html=True)
    st.markdown("<span style='color: #9ca3af; font-size: 12px;'>VENIAM (82%)</span>", unsafe_allow_html=True)
    st.progress(0.82)
    st.markdown("<span style='color: #9ca3af; font-size: 12px;'>CULLUM (65%)</span>", unsafe_allow_html=True)
    st.progress(0.65)
    st.markdown("<span style='color: #9ca3af; font-size: 12px;'>LABORE (91%)</span>", unsafe_allow_html=True)
    st.progress(0.91)

with col_m2:
    st.markdown("""<div class="card"><div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">MINIM — Telemetry Spectrum</div><div style="font-size: 22px; font-weight: bold; color: #ffffff;">476</div></div>""", unsafe_allow_html=True)
    x_w = np.arange(30)
    y_w = np.sin(x_w * 0.4) * np.exp(-x_w * 0.02) + np.random.normal(0, 0.1, 30)
    fig_w = go.Figure(go.Scatter(x=x_w, y=y_w, mode='lines', line=dict(color='#a855f7', width=2), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.15)'))
    fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=5, b=5, l=5, r=5), height=110, font_color="#e5e7eb", xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_w, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("""<div class="card"><div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">LOREM — Risk Breakdown</div></div>""", unsafe_allow_html=True)
    counts = df.groupby("Category")["ID"].count().reset_index()
    fig_donut = px.pie(counts, names="Category", values="ID", hole=0.65, color_discrete_sequence=["#38bdf8", "#ec4899", "#facc15"])
    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), height=180, font_color="#e5e7eb", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_donut, use_container_width=True)

with g2:
    st.markdown("""<div class="card"><div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">IRURE — ECE Calibration Ring</div></div>""", unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=180347,
        number={'prefix': "$", 'font': {'size': 18, 'color': '#ffffff'}},
        gauge={'axis': {'range': [0, 200000]}, 'bar': {'color': '#38bdf8'}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0, 'steps': [{'range': [0, 200000], 'color': '#1f2437'}]}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=10, l=10, r=10), height=160, font_color="#e5e7eb")
    st.plotly_chart(fig_gauge, use_container_width=True)

with g3:
    st.markdown("""<div class="card"><div style="font-size: 14px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">MAGNA — Noise Stress Test</div></div>""", unsafe_allow_html=True)
    x_t = np.arange(40)
    y_t = np.random.uniform(0.2, 0.9, 40)
    fig_bar = px.bar(x=x_t, y=y_t, color_discrete_sequence=["#38bdf8"])
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=5, b=5, l=5, r=5), height=170, font_color="#e5e7eb", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
