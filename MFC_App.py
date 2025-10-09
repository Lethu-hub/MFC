import streamlit as st
from utils.data_loader import load_all_data

st.set_page_config(page_title="MFC Home", layout="wide")
st.title("🏟️ Welcome to MFC - Mighty Football Club")
st.subheader("Explore our history, performance, and predictions!")

# -----------------------------
# Load datasets
# -----------------------------
dfs = load_all_data()
players_df = dfs["Players"]
matches_df = dfs["Matches"]
events_df = dfs["Match Events"]

# -----------------------------
# Club Summary
# -----------------------------
st.markdown("""
**About MFC:**  
Founded in 1998, MFC has been competing in national and international competitions for over 25 years.  
Our mission: develop world-class talent and dominate on the pitch.
""")

# Basic Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Seasons Played", matches_df['Season'].nunique())
col2.metric("Total Matches", matches_df.shape[0])
col3.metric("Players Registered", players_df.shape[0])
col4.metric("Total Events Recorded", events_df.shape[0])

st.markdown("---")

# -----------------------------
# Navigation Links
# -----------------------------
st.subheader("Explore Our Data")

st.markdown("""
Use the buttons below to navigate to different pages:
""")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📜 History"):
        st.experimental_set_query_params(page="history")

with col2:
    if st.button("📈 Performance Analysis"):
        st.experimental_set_query_params(page="performance")

with col3:
    if st.button("⚡ Predictions"):
        st.experimental_set_query_params(page="predictions")

st.markdown("---")

# -----------------------------
# Upcoming Match Preview (Optional)
# -----------------------------
st.subheader("Upcoming Match")
upcoming_match = matches_df[matches_df['Date'] > pd.Timestamp.today()].sort_values('Date').head(1)
if not upcoming_match.empty:
    match = upcoming_match.iloc[0]
    st.markdown(f"**{match['Home_Team']} vs {match['Away_Team']}**")
    st.markdown(f"Date: {match['Date']}")
    st.markdown(f"Competition: {match['Competition']}")
else:
    st.markdown("No upcoming matches recorded.")

