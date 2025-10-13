import streamlit as st
import pandas as pd
from data_loader import load_all_data

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="MFC Dashboard", layout="wide")

# ==========================
# Landing Page Header
# ==========================
st.markdown(
    "<h1 style='text-align: center; color: #0055A4;'>Welcome to MFC Dashboard 🏆</h1>",
    unsafe_allow_html=True
)

# ==========================
# Banner Image (Optional)
# ==========================
# st.image("mfc_banner.jpg", use_container_width=True)  # Make sure the image exists in your repo

# ==========================
# Load Data
# ==========================
dfs = load_all_data()
matches_df = dfs.get("Matches", pd.DataFrame())

# ==========================
# Upcoming Matches Section
# ==========================
st.subheader("Upcoming Matches")
if not matches_df.empty:
    # Ensure date column exists and is datetime
    matches_df['Match_Date'] = pd.to_datetime(matches_df['Match_Date'], errors='coerce')
    upcoming_matches = matches_df[matches_df['Match_Date'] > pd.Timestamp.today()]
    
    if not upcoming_matches.empty:
        # Show top 3 upcoming matches
        for i, match in upcoming_matches.head(3).iterrows():
            st.markdown(f"**{match['HomeTeam']} vs {match['AwayTeam']}** on {match['Match_Date'].strftime('%Y-%m-%d')} at {match.get('Venue', 'Unknown')}")
    else:
        st.info("No upcoming matches scheduled.")
else:
    st.warning("Matches data not loaded.")
