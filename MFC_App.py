import streamlit as st
import pandas as pd
from datetime import datetime

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="🏟️ MFC Home", layout="wide")

# ==========================
# Sidebar - retractable menu simulation
# ==========================
with st.sidebar:
    st.title("🏟️ MFC Menu")
    show_home = st.button("🏠 Home")
    show_history = st.button("📊 Historical Data")
    show_performance = st.button("📈 Performance Analysis")
    show_predictions = st.button("🎯 Predictions")

# ==========================
# Banner Image
# ==========================
st.image("mfc_banner.png", use_column_width=True)  # Replace with your banner image path

# ==========================
# Load matches
# ==========================
@st.cache_data
def load_upcoming_matches():
    df = pd.read_csv("matches.csv")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df

matches_df = load_upcoming_matches()

# Filter future matches
today = datetime.today().date()
future_matches = matches_df[matches_df["Date"].dt.date >= today].sort_values("Date")
next_three = future_matches.head(3)

# ==========================
# Page Content
# ==========================
if show_home or (not show_history and not show_performance and not show_predictions):
    st.title("🏟️ Welcome to Mighty Football Club (MFC)")
    st.markdown("#### ⚽ Founded in 2022 — Champions in Passion, Unity & Performance")

    st.subheader("📅 Next 3 Upcoming Matches")
    
    if not next_three.empty:
        for i, match in next_three.iterrows():
            match_date = match['Date']
            days_left = (match_date.date() - today).days

            # Use columns for a card-like display
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"**{match['HomeTeam']} vs {match['AwayTeam']}**")
                st.markdown(f"**Competition:** {match['Competition']}")
            with col2:
                st.markdown(f"📅 {match_date.date()}")
                st.markdown(f"⏱️ Kickoff: {match['KickOffTime']}")
            with col3:
                st.markdown(f"📍 {match['Venue']}")
                st.markdown(f"⏳ {days_left} days left")
            st.markdown("---")
    else:
        st.info("✅ No upcoming matches — the season might be complete!")

elif show_history:
    st.title("📊 Historical Performance")
    st.markdown("Explore historical match data, univariate & bivariate analysis.")

elif show_performance:
    st.title("📈 Performance Analysis")
    st.markdown("Player and team performance analytics, charts, heatmaps, violin plots, etc.")

elif show_predictions:
    st.title("🎯 Predictions")
    st.markdown("Upcoming match predictions, yellow/red cards, goal scorers, etc.")
