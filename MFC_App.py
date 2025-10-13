import streamlit as st
import pandas as pd

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="MFC Dashboard", layout="wide")

# ==========================
# Sidebar with Collapsible Menu
# ==========================
with st.sidebar.expander("🏟️ MFC Menu"):
    page = st.radio("Navigate to:", ["Home", "Data Exploration", "Performance Analysis", "Predictions"])

# ==========================
# Banner
# ==========================
st.image("mfc_banner.jpg", use_container_width=True)  # Replace with your banner image

# ==========================
# Home Page
# ==========================
if page == "Home":
    st.title("Welcome to MFC Dashboard")
    st.subheader("Your football data hub")
    st.markdown("""
    MFC is a hub to explore football stats, performances, and predictions. 
    Check out historical data or see what’s coming up next!
    """)

    # ==========================
    # Load Matches
    # ==========================
    matches_df = pd.read_csv("matches.csv")
    matches_df["Match_Date"] = pd.to_datetime(matches_df["Match_Date"], errors='coerce')

    # Filter upcoming matches
    today = pd.Timestamp.today()
    upcoming_matches = matches_df[matches_df["Match_Date"] >= today].sort_values("Match_Date").head(3)

    st.subheader("Upcoming Matches")
    if not upcoming_matches.empty:
        for idx, row in upcoming_matches.iterrows():
            st.markdown(f"**{row['HomeTeam']} vs {row['AwayTeam']}**")
            st.markdown(f"📅 Date: {row['Match_Date'].date()} | 🕒 KickOff: {row['KickOffTime']}")
            st.markdown("---")
    else:
        st.info("No upcoming matches scheduled yet!")

# ==========================
# Data Exploration Page Placeholder
# ==========================
elif page == "Data Exploration":
    st.header("Data Exploration & Univariate Analysis")
    st.write("Load and explore historical data here...")

# ==========================
# Performance Analysis Page Placeholder
# ==========================
elif page == "Performance Analysis":
    st.header("Performance Analysis")
    st.write("Bivariate and multivariate charts go here...")

# ==========================
# Predictions Page Placeholder
# ==========================
elif page == "Predictions":
    st.header("Predictions & Upcoming Matches")
    st.write("Predict goals, cards, or player stats for upcoming matches...")
