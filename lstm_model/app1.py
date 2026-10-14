import streamlit as st
import pandas as pd
import os
import altair as alt

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>

/* Page background */

/* Main container spacing */
.block-container {
    padding-top: 1.5rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #eef3ff;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

/* Generic card */
.card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
/* Status cards (pastel variants) */
.status-good {
    background-color: #eafaf1;
}
.status-satisfactory {
    background-color: #fef9e7;
}
.status-moderate {
    background-color: #fdf2e9;
}
.status-poor {
    background-color: #fdecea;
}
.status-very-poor {
    background-color: #f4ecf7;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("AQI Dashboard")

city = st.sidebar.selectbox(
    "Select City",
    ["delhi", "mumbai", "bengaluru"],
    format_func=lambda x: x.capitalize()
)

st.sidebar.markdown("### AQI Scale")
st.sidebar.markdown("""
- 🟢 0–50 : Good  
- 🟡 51–100 : Satisfactory  
- 🟠 101–200 : Moderate  
- 🔴 201–300 : Poor  
- 🟣 301+ : Very Poor
""")

csv_path = os.path.join("model", f"{city}_predictions.csv")

if not os.path.exists(csv_path):
    st.error("Prediction file not found.")
    st.stop()

with st.spinner("Loading AQI predictions..."):
    df = pd.read_csv(csv_path)

latest_aqi = int(df["predicted_aqi"].iloc[-1])

def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    else:
        return "Very Poor"

def aqi_color(aqi):
    if aqi <= 50:
        return "#2ecc71"
    elif aqi <= 100:
        return "#f1c40f"
    elif aqi <= 200:
        return "#e67e22"
    elif aqi <= 300:
        return "#e74c3c"
    else:
        return "#9b59b6"
    
status_text = aqi_category(latest_aqi)
line_color = aqi_color(latest_aqi)

status_class = {
    "Good": "status-good",
    "Satisfactory": "status-satisfactory",
    "Moderate": "status-moderate",
    "Poor": "status-poor",
    "Very Poor": "status-very-poor"
}[status_text]

st.title("Air Quality Index (AQI) Forecast")
st.caption("LSTM-based AQI Prediction")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📍 City", city.capitalize())
c2.metric("Latest AQI", latest_aqi)
c3.metric("Min AQI", int(df["predicted_aqi"].min()))
c4.metric("Max AQI", int(df["predicted_aqi"].max()))

st.markdown(
    f"""
    <div class="card {status_class}">
        <h3 style="margin:0;">
            Current Status:
            <span style="color:{line_color}; font-weight:600;">
                {status_text}
            </span>
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("AQI Trend")

aqi_chart = (
    alt.Chart(df)
    .mark_line(
        strokeWidth=3,
        color=line_color
    )
    .encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("predicted_aqi:Q", title="AQI"),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("predicted_aqi:Q", title="AQI")
        ]
    )
    .properties(height=360)
    .interactive()
)
st.altair_chart(aqi_chart, use_container_width=True)

with st.expander("View Detailed AQI Data"):
    display_df = df.rename(columns={
        "timestamp": "Time",
        "predicted_aqi": "Predicted AQI"
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Forecast CSV",
    df.to_csv(index=False).encode("utf-8"),
    f"{city}_predictions.csv",
    "text/csv"
)
st.caption("AQI Forecast Dashboard • Streamlit")
