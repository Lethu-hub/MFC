
import pandas as pd
import streamlit as st

# ==========================
# 🚀 Load CSVs into Pandas
# ==========================
@st.cache_data
def load_all_data():
    dfs = {}
    for name, path in DATASETS.items():
        try:
            print(f"Loading {name} from {path}...")
            dfs[name] = pd.read_csv(path)
            print(f"✅ {name}: {dfs[name].shape[0]:,} rows × {dfs[name].shape[1]:,} columns")
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
    return dfs
# ==========================
# 📁 CSV paths (relative to repo)
# ==========================
DATASETS = {
    "Players": "players.csv",
    "Matches": "matches.csv",
    "Match Events": "match_events.csv"
}
