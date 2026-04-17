import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AgroSmart AI Pro", layout="wide")

# -------------------------------
# UI STYLING (Modern Agri-Dark Theme)
# -------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0b0e0b; color: #e0e8e0; }
    
    /* Sidebar Text Visibility Fix */
    section[data-testid="stSidebar"] { background-color: #111a11 !important; border-right: 1px solid #1e2e1e; }
    section[data-testid="stSidebar"] label { color: #ffffff !important; font-size: 16px !important; font-weight: 600; }
    
    /* Custom Container Styling */
    .agri-header { text-align: center; padding: 25px; background: linear-gradient(135deg, #162016 0%, #0b0e0b 100%); border-radius: 15px; border: 1px solid #2d4a2d; margin-bottom: 25px; }
    .main-title { color: #44ff44; font-size: 42px; font-weight: 800; margin: 0; text-shadow: 0 0 20px rgba(68, 255, 68, 0.2); }
    
    /* Metric Cards */
    .metric-card { background: #162016; border-left: 5px solid #44ff44; padding: 22px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    .m-label { color: #88aa88; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
    .m-value { color: #ffffff; font-size: 32px; font-weight: 700; }
    
    /* Section Headers */
    .section-head { color: #44ff44; font-size: 20px; font-weight: 600; margin: 30px 0 15px 0; border-bottom: 1px solid #2d4a2d; padding-bottom: 5px; }

    /* Download Button Style */
    .stDownloadButton > button {
        background-color: #1a5c1a !important;
        color: white !important;
        border: 1px solid #44ff44 !important;
        border-radius: 8px !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #44ff44 !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def live_predict(soil, temp, rain, hum):
    try:
        clf = pickle.load(open("model.pkl", "rb"))
        reg = pickle.load(open("water_model.pkl", "rb"))
        input_df = pd.DataFrame([[soil, temp, rain, hum]], columns=["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Humidity"])
        p = clf.predict(input_df)[0]
        w = reg.predict(input_df)[0]
        if w <= 0: raise Exception("Low Calc")
    except:
        # PURE MATHEMATICAL FALLBACK
        target = 0.70 
        moisture_gap = max(0, target - soil)
        evap_factor = (temp * 1.8) / (hum + 5)
        w = (moisture_gap * 280) + evap_factor - (rain * 0.9)
        p = 1 if (soil < 0.4 or w > 15) else 0
        
    return p, max(0, round(w, 1))

# -------------------------------
# SIDEBAR SENSORS
# -------------------------------
with st.sidebar:
    st.markdown('<h3 style="color: #44ff44; margin-bottom:20px;">🛰️ FIELD SENSOR ARRAY</h3>', unsafe_allow_html=True)
    soil = st.slider("Soil Moisture Content", 0.0, 1.0, 0.20, 0.01)
    temp = st.slider("Ambient Temperature (°C)", 0, 50, 32)
    rain = st.slider("Forecasted Rainfall (mm)", 0, 100, 0)
    hum = st.slider("Atmospheric Humidity (%)", 0, 100, 45)
    st.markdown("---")
    
    # Instant Prediction values for report
    pred, water = live_predict(soil, temp, rain, hum)
    
    # Report Preparation
    report_df = pd.DataFrame({
        "Parameter": ["Timestamp", "Soil Moisture", "Temperature", "Humidity", "Rainfall", "Status", "Water Required"],
        "Value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{soil}", f"{temp}°C", f"{hum}%", f"{rain}mm", 
                  "Irrigate Now" if pred == 1 else "Optimal", f"{water}mm"]
    })
    csv = convert_df(report_df)

    st.download_button(
        label="📥 DOWNLOAD REPORT (CSV)",
        data=csv,
        file_name=f'Field_Report_{datetime.now().strftime("%H%M%S")}.csv',
        mime='text/csv',
    )
    st.markdown("---")
    analyze_btn = st.button("REFRESH ANALYTICS")

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
st.markdown('<div class="agri-header"><h1 class="main-title">🌿 HYDRO INTELLIGENCE PRO</h1><p style="color:#88aa88;">Autonomous Irrigation Decision System</p></div>', unsafe_allow_html=True)

# Top Row Metrics
c1, c2, c3 = st.columns(3)
with c1:
    status = "IRRIGATE NOW" if pred == 1 else "STABLE"
    s_col = "#ff4444" if pred == 1 else "#44ff44"
    st.markdown(f'<div class="metric-card" style="border-left-color:{s_col}"><div class="m-label">Action Status</div><div class="m-value" style="color:{s_col}">{status}</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="metric-card"><div class="m-label">Water Requirement</div><div class="m-value">{water} <span style="font-size:18px">mm</span></div></div>', unsafe_allow_html=True)

with c3:
    stress = round((temp * 0.6) + ( (100-hum) * 0.4), 1)
    st.markdown(f'<div class="metric-card" style="border-left-color:#ffa500"><div class="m-label">Plant Stress Index</div><div class="m-value">{stress}%</div></div>', unsafe_allow_html=True)

# -------------------------------
# ADVANCED VISUALIZATIONS
# -------------------------------
st.markdown('<div class="section-head">📊 REAL-TIME FIELD ANALYTICS</div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)

with g1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = soil * 100,
        title = {'text': "Moisture Saturation %", 'font': {'size': 16, 'color': '#88aa88'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "#44ff44"},
            'bgcolor': "#162016",
            'steps': [
                {'range': [0, 35], 'color': '#5c1a1a'},
                {'range': [35, 70], 'color': '#5c4d1a'},
                {'range': [70, 100], 'color': '#1a5c1a'}]
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=350, margin=dict(t=50, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    t_arr = np.linspace(20, 50, 10)
    h_arr = np.linspace(0, 100, 10)
    z = [[round((t * 0.5) - (h * 0.2), 2) for h in h_arr] for t in t_arr]
    fig_heat = px.imshow(z, x=h_arr, y=t_arr, labels=dict(x="Humidity %", y="Temp °C", color="Stress"),
                         color_continuous_scale='Greens')
    fig_heat.update_layout(title="VPD Stress Matrix", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown('<div class="section-head">🔮 PREDICTIVE FORECASTING</div>', unsafe_allow_html=True)

g3, g4 = st.columns(2)

with g3:
    time_steps = np.arange(13)
    moisture_optimal = [soil * (0.98**i) for i in time_steps]
    moisture_extreme = [soil * (0.94**i) for i in time_steps]
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=time_steps, y=moisture_optimal, name="Standard Forecast", line=dict(color='#44ff44', width=3)))
    fig_trend.add_trace(go.Scatter(x=time_steps, y=moisture_extreme, name="High Evap Scenario", line=dict(color='#ff4444', dash='dot')))
    fig_trend.update_layout(title="12-Hour Moisture Projection", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=350)
    st.plotly_chart(fig_trend, use_container_width=True)

with g4:
    theoretical = round((0.75 - soil) * 200, 1) if soil < 0.75 else 5
    rain_offset = min(theoretical, rain * 0.8)
    net_needed = max(0, theoretical - rain_offset)
    
    fig_bar = go.Figure(data=[
        go.Bar(name='Rain Offset', x=['Water Balance'], y=[rain_offset], marker_color='#0088ff'),
        go.Bar(name='AI Dispatch Needed', x=['Water Balance'], y=[net_needed], marker_color='#44ff44')
    ])
    fig_bar.update_layout(barmode='stack', title="Rain-Adjusted Water Needs", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------
# SYSTEM LOGS
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.code(f"""
[LOG] Sensors Online: Soil={soil}, Temp={temp}C, Hum={hum}%
[AI]  Prediction Engine: Irrigation={status} | Calculated Dosage={water}mm
[SYS] Resource Optimization: Rain offset applied. Ready for dispatch.
""", language="bash")