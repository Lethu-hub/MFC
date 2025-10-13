import streamlit as st
import pandas as pd
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="🏟️ MFC Home", layout="wide")

# ---------------------------
# Sidebar - retractable menu simulation
# ---------------------------
with st.sidebar:
    st.title("🏟️ MFC Menu")
    show_home = st.button("🏠 Home")
    show_history = st.button("📊 Historical Data")
    show_performance = st.button("📈 Performance Analysis")
    show_predictions = st.button("🎯 Predictions")
    
# ---------------------------
# Load upcoming matches
# ---------------------------
@st.cache_data
def load_upcoming_matches():
    df = pd.read_csv("upcoming_matches.csv")
    df.columns = df.columns.str.strip()  # Remove hidden spaces
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df

matches_df = load_upcoming_matches()

today = datetime.today().date()
future_matches = matches_df[matches_df["Date"].dt.date >= today].sort_values("Date")

# ---------------------------
# Page content based on sidebar buttons
# ---------------------------
if show_home or (not show_history and not show_performance and not show_predictions):
    st.title("🏟️ Welcome to Mighty Football Club (MFC)")
    st.markdown("#### ⚽ Founded in 2022 — Champions in Passion, Unity & Performance")

    st.subheader("📅 Upcoming Matches")
    if not future_matches.empty:
        display_cols = ["Date", "KickOffTime", "HomeTeam", "AwayTeam", "Competition", "Venue", "Weather"]
        st.dataframe(future_matches[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("✅ No upcoming matches — the season might be complete!")

elif show_history:
    st.title("📊 Historical Performance")
    st.markdown("This is where we can explore historical match data, univariate & bivariate analysis.")

elif show_performance:
    st.title("📈 Performance Analysis")
    st.markdown("Here we show player and team performance analytics, charts, heatmaps, violin plots, etc.")

elif show_predictions:
    st.title("🎯 Predictions")
    st.markdown("Upcoming match predictions, yellow/red cards, goal scorers, etc.")
