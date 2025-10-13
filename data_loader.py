import streamlit as st
import pandas as pd
from datetime import datetime

# ==========================
# ⚙️ Page Config
# ==========================
st.set_page_config(page_title="🏟️ MFC Home", layout="wide")

st.title("🏟️ Welcome to Mighty Football Club (MFC)")
st.markdown("#### ⚽ Founded in 2022 — Champions in Passion, Unity & Performance")

# ==========================
# 📂 Load Data
# ==========================
@st.cache_data
def load_upcoming_matches():
    try:
        df = pd.read_csv("upcoming_matches.csv", parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error("⚠️ Could not find 'upcoming_matches.csv'. Please generate it first.")
        return pd.DataFrame()

upcoming_df = load_upcoming_matches()

# ==========================
# 📅 Filter: Only Future Matches
# ==========================
today = datetime.today().date()
future_matches = upcoming_df[upcoming_df["Date"].dt.date >= today].sort_values("Date")

st.markdown("## 📅 Upcoming Matches")
if not future_matches.empty:
    # Show only key columns neatly formatted
    display_cols = ["Date", "KickOffTime", "HomeTeam", "AwayTeam", "Competition", "Venue", "Weather"]
    st.dataframe(
        future_matches[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("✅ No upcoming matches — the season might be complete!")

# ==========================
# 🔗 Navigation Links
# ==========================
st.markdown("---")
st.markdown("### 🔍 Explore More")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View Historical Performance"):
        st.switch_page("pages/history.py")

with col2:
    if st.button("📈 Performance Analysis"):
        st.switch_page("pages/performance.py")

with col3:
    if st.button("🎯 Predictions"):
        st.switch_page("pages/predictions.py")
