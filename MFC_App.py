import pandas as pd
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="MFC Dashboard", layout="wide")
st.title("🏟️ Welcome to MFC Dashboard")

# -------------------------
# Load matches CSV
# -------------------------
matches_df = pd.read_csv("matches.csv")

# Ensure dates are datetime
matches_df['Match_Date'] = pd.to_datetime(matches_df['Match_Date'], errors='coerce')

# -------------------------
# Upcoming Matches (next 3)
# -------------------------
today = pd.Timestamp.today()
upcoming_matches = matches_df[matches_df['Match_Date'] > today].sort_values('Match_Date')

st.subheader("Upcoming Matches")
if upcoming_matches.empty:
    st.info("No upcoming matches scheduled yet!")
else:
    for _, match in upcoming_matches.head(3).iterrows():
        st.markdown(f"**{match['HomeTeam']} vs {match['AwayTeam']}**")
        st.markdown(f"📅 Date: {match['Match_Date'].date()}  |  🏟️ Venue: {match['Venue']}  |  ⚽ Competition: {match['Competition']}")
        st.markdown("---")
