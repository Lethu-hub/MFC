# data_loader.py
import pandas as pd
from datetime import datetime

def load_upcoming_matches():
    """Load upcoming matches CSV."""
    try:
        df = pd.read_csv("upcoming_matches.csv", parse_dates=["Date"])
        return df
    except FileNotFoundError:
        print("⚠️ Could not find 'upcoming_matches.csv'.")
        return pd.DataFrame()

def load_all_data():
    """Load all main CSV datasets."""
    try:
        players_df = pd.read_csv("players.csv")
        matches_df = pd.read_csv("matches.csv")
        events_df = pd.read_csv("match_events.csv")
        return {
            "Players": players_df,
            "Matches": matches_df,
            "Match Events": events_df
        }
    except FileNotFoundError as e:
        print(f"⚠️ Missing file: {e}")
        return {}
