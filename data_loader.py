import pandas as pd
import streamlit as st

# ==========================
# CSV file names
# ==========================
PLAYERS_PATH = "players.csv"
MATCHES_PATH = "matches.csv"
EVENTS_PATH = "match_events.csv"

# ==========================
# Load CSVs into Pandas
# ==========================
@st.cache_data
def load_all_data():
    dfs = {}
    
    try:
        dfs["Players"] = pd.read_csv(PLAYERS_PATH)
        print(f"✅ Players: {dfs['Players'].shape}")
    except Exception as e:
        print(f"⚠️ Failed to load Players: {e}")
        
    try:
        dfs["Matches"] = pd.read_csv(MATCHES_PATH)
        print(f"✅ Matches: {dfs['Matches'].shape}")
    except Exception as e:
        print(f"⚠️ Failed to load Matches: {e}")
        
    try:
        dfs["Match Events"] = pd.read_csv(EVENTS_PATH)
        print(f"✅ Match Events: {dfs['Match Events'].shape}")
    except Exception as e:
        print(f"⚠️ Failed to load Match Events: {e}")
        
    return dfs
