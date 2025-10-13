import pandas as pd
import streamlit as st

# ==========================
# Load matches CSV
# ==========================
@st.cache_data
def load_matches():
    df = pd.read_csv("matches.csv")
    df.columns = df.columns.str.strip()
    df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors='coerce')
    return df

matches_df = load_matches()

# ==========================
# Page Setup
# ==========================
st.set_page_config(page_title="MFC Home", layout="wide")

# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.title("🏟️ MFC Menu")
page = st.sidebar.radio("Navigate:", ["Home", "History", "Performance", "Predictions"])

# ==========================
# Home / Landing Page
# ==========================
if page == "Home":
    # Banner image
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3b/Football_pitch.png",
             use_container_width=True)
    
    st.markdown("<h1 style='text-align: center; color: #0055A4;'>Welcome to Mighty Football Club 🏆</h1>", unsafe_allow_html=True)
    st.markdown("""
    Mighty Football Club (MFC) was founded in [year].  
    We have played countless matches and continue to strive for excellence.  
    """)

    # ==========================
    # Upcoming Matches
    # ==========================
    st.subheader("⚡ Upcoming Matches")
    upcoming_matches = matches_df[matches_df["Match_Date"] > pd.Timestamp.today()]
    upcoming_matches = upcoming_matches.sort_values("Match_Date").head(3)

    if not upcoming_matches.empty:
        for idx, match in upcoming_matches.iterrows():
            # Default colors/icons (replace with team logos if available)
            home_color = "#0055A4"
            away_color = "#D32F2F"
            yellow_cards = match.get("Home_Yellow", 0)
            red_cards = match.get("Home_Red", 0)

            st.markdown(f"""
            <div style='border:2px solid {home_color}; padding:15px; border-radius:10px; margin-bottom:10px; transition: all 0.3s;'>
                <h3 style='color:{home_color};'>{match['HomeTeam']} vs <span style='color:{away_color};'>{match['AwayTeam']}</span></h3>
                <p><strong>Date:</strong> {match['Match_Date'].strftime('%Y-%m-%d')} | <strong>Season:</strong> {match['Season']}</p>
                <p><strong>Venue:</strong> {match.get('Venue', 'N/A')} | <strong>Competition:</strong> {match.get('Competition', 'N/A')}</p>
                <p>🟨 {yellow_cards} | 🟥 {red_cards}</p>
                <button onclick="alert('Navigate to Predictions!')">View Prediction</button>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No upcoming matches scheduled.")

    # ==========================
    # Quick Action Buttons
    # ==========================
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📜 View History"):
            st.session_state.page = "History"
    with col2:
        if st.button("📊 Performance Analysis"):
            st.session_state.page = "Performance"
    with col3:
        if st.button("🎯 Make Predictions"):
            st.session_state.page = "Predictions"
