import streamlit as st
import pandas as pd
from data_loader import load_all_data
from stats import central_tendency, measures_of_spread, categorical_counts
from charts import univariate_chart
from datetime import datetime

# ==========================
# Page config
# ==========================
st.set_page_config(
    page_title="MFC Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Statistics", "Performance", "Predictions"])

# ==========================
# Home Page
# ==========================
if page == "Home":
    st.title("🏟️ Welcome to MFC Dashboard")
    st.subheader("Upcoming Matches")
    try:
        upcoming_matches_df = pd.read_csv("upcoming_matches.csv")
        upcoming_matches_df['Date'] = pd.to_datetime(upcoming_matches_df['Date'], errors='coerce')
        upcoming_matches_df = upcoming_matches_df[upcoming_matches_df['Date'] >= pd.Timestamp.today()]
    except FileNotFoundError:
        st.error("upcoming_matches.csv not found!")
        upcoming_matches_df = pd.DataFrame()

    if upcoming_matches_df.empty:
        st.info("No upcoming matches scheduled yet!")
    else:
        upcoming_matches_df = upcoming_matches_df.sort_values(by='Date')
        for _, match in upcoming_matches_df.head(3).iterrows():
            st.markdown(
                f"""
                <div style="border:1px solid #e1e1e1; padding:15px; border-radius:10px; margin-bottom:10px; background-color:#f9f9f9;">
                    <h4 style="margin:0;">{match['HomeTeam']} vs {match['AwayTeam']}</h4>
                    <p style="margin:0;">📅 Date: {match['Date'].strftime('%A, %d %B %Y')}<br>
                    🏟️ Venue: {match['Venue']}<br>
                    ⚽ Competition: {match['Competition']}</p>
                </div>
                """, unsafe_allow_html=True
            )
elif page == "Statistics":
    st.title("📊 MFC History & Stats")
    st.markdown("Explore player and match data so far — Data Exploration & Stats")

    from data_loader import load_all_data

    # -----------------------------
    # Load datasets
    # -----------------------------
    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    # Merge player names for easier analysis
    df = events_df.merge(players_df, on="Player_ID")

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.sidebar.header("Filters")
    season_filter = st.sidebar.multiselect(
        "Select Season", options=df['Season'].unique(), default=df['Season'].unique()
    )
    player_filter = st.sidebar.multiselect(
        "Select Player", options=players_df['Player_Name'].unique(), default=players_df['Player_Name'].unique()
    )
    event_filter = st.sidebar.multiselect(
        "Select Event Type", options=df['Event_Type'].unique(), default=df['Event_Type'].unique()
    )
    match_filter = st.sidebar.selectbox(
        "Select Match ID (optional for team stats)", options=[None]+df['Match_ID'].unique().tolist()
    )

    # Apply filters
    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # -----------------------------
    # Tabs
    # -----------------------------
    tab_explore, tab_player, tab_team = st.tabs(["Data Exploration", "Player Stats", "Team Stats"])

    # -----------------------------
    # Data Exploration Tab
    # -----------------------------
    with tab_explore:
        st.subheader("Raw Data")
        st.dataframe(df_filtered, use_container_width=True)

    # -----------------------------
    # Player Stats Tab
    # -----------------------------
    with tab_player:
        st.subheader("Player-Level Summary")
        if df_filtered.empty:
            st.info("No player data available for selected filters.")
        else:
            player_summary = df_filtered.groupby("Player_Name").agg(
                Matches_Played=("Match_ID", "nunique"),
                Goals=lambda x: (x=="Goal").sum(),
                Assists=lambda x: (x=="Assist").sum(),
                Yellow_Cards=lambda x: (x=="Yellow Card").sum(),
                Red_Cards=lambda x: (x=="Red Card").sum(),
                Fouls=lambda x: (x=="Foul").sum()
            ).reset_index()

            st.dataframe(player_summary.sort_values(by="Matches_Played", ascending=False), use_container_width=True)

    # -----------------------------
    # Team Stats Tab
    # -----------------------------
    with tab_team:
        st.subheader("Team-Level Summary (Selected Match)")
        if match_filter is None:
            st.info("Select a Match ID to see team-level stats.")
        else:
            team_df = df[df['Match_ID'] == match_filter]

            # Apply event filter
            if event_filter:
                team_df = team_df[team_df['Event_Type'].isin(event_filter)]

            if team_df.empty:
                st.warning("No events match the selected filters for this match.")
            else:
                team_summary = team_df.groupby("Match_ID").agg(
                    Goals=("Event_Type", lambda x: (x=="Goal").sum()),
                    Assists=("Event_Type", lambda x: (x=="Assist").sum()),
                    Yellow_Cards=("Event_Type", lambda x: (x=="Yellow Card").sum()),
                    Red_Cards=("Event_Type", lambda x: (x=="Red Card").sum()),
                    Fouls=("Event_Type", lambda x: (x=="Foul").sum())
                ).reset_index()

                st.dataframe(team_summary, use_container_width=True)

                st.markdown("**Players in this match:**")
                st.dataframe(
                    team_df[["Player_Name", "Event_Type", "Minute"]].sort_values(by="Minute"),
                    use_container_width=True
                )

# ==========================
# Performance Page
# ==========================
elif page == "Performance":
    st.title("📈 Team & Player Performance")
    # Logic similar to History page; you can reuse the tabs with team stats included
    # Could include season-wide aggregates, player vs team comparison, etc.
    # For brevity, same structure as History with Team Tab included

# ==========================
# Predictions Page
# ==========================
elif page == "Predictions":
    st.title("⚡ MFC Player Predictions: Upcoming Matches")

    @st.cache_data
    def load_data():
        try:
            players_df = pd.read_csv("players.csv")
            events_df = pd.read_csv("match_events.csv")
        except FileNotFoundError:
            st.error("⚠️ Missing core datasets.")
            return None, None, pd.DataFrame()
        try:
            upcoming_df = pd.read_csv("upcoming_matches.csv", parse_dates=["Date"])
        except FileNotFoundError:
            st.warning("⚠️ 'upcoming_matches.csv' not found.")
            upcoming_df = pd.DataFrame()
        return players_df, events_df, upcoming_df

    players_df, events_df, upcoming_df = load_data()
    if players_df is None or events_df is None:
        st.stop()

    # Upcoming matches
    if upcoming_df.empty:
        st.info("✅ No upcoming matches available.")
    else:
        mfc_upcoming = upcoming_df[
            (upcoming_df["HomeTeam"].str.upper() == "MFC") |
            (upcoming_df["AwayTeam"].str.upper() == "MFC")
        ]
        if not mfc_upcoming.empty:
            st.subheader("📅 Upcoming MFC Matches")
            st.dataframe(
                mfc_upcoming[["Date", "HomeTeam", "AwayTeam", "Competition", "Venue"]],
                hide_index=True, use_container_width=True
            )

    # Team stats tab for Predictions page
    st.subheader("🏆 Team-Level Stats for Selected Match")
    if not upcoming_df.empty:
        upcoming_match_ids = mfc_upcoming['Match_ID'].unique()
        selected_match_pred = st.selectbox("Select Upcoming Match", upcoming_match_ids)
        team_df_pred = events_df[events_df['Match_ID'] == selected_match_pred]
        match_players_pred = team_df_pred.merge(players_df, on='Player_ID')['Player_Name'].unique()
        st.multiselect("Players in Match", match_players_pred, default=match_players_pred, disabled=True)
        event_filter_team_pred = st.multiselect(
            "Filter Event Type", options=team_df_pred['Event_Type'].unique(), default=team_df_pred['Event_Type'].unique()
        )
        team_df_filtered_pred = team_df_pred[team_df_pred['Event_Type'].isin(event_filter_team_pred)]
        team_stats_pred = team_df_filtered_pred.groupby('Event_Type').size().reset_index(name='Count')
        st.table(team_stats_pred)

# ==========================
# Predictions Page
# ==========================
elif page == "Predictions":
    st.title("⚡ MFC Player Predictions: Upcoming Matches")

    from data_loader import load_all_data
    dfs = load_all_data()
    players_df = dfs["Players"]
    events_df = dfs["Match Events"]

    try:
        upcoming_df = pd.read_csv("upcoming_matches.csv", parse_dates=["Date"])
    except FileNotFoundError:
        st.warning("No upcoming matches found")
        upcoming_df = pd.DataFrame()

    numeric_cols = events_df.select_dtypes(include="number").columns.tolist()
    target_col = st.sidebar.selectbox("🎯 Target Variable", numeric_cols)
    feature_cols = st.sidebar.multiselect("🧩 Features", [c for c in events_df.columns if c != target_col])

    tab1, tab_team = st.tabs(["Player Predictions", "🏆 Team Predictions"])

    # Tab1: Player Predictions
    with tab1:
        st.write("⚡ Predictions per player (existing code here)")

    # Tab2: Team Predictions
    with tab_team:
        if not upcoming_df.empty:
            match_id = st.selectbox("Select Upcoming Match", upcoming_df.index, key="team_pred")
            match_players = players_df  # Could merge historical stats or use averages
            numeric_cols = events_df.select_dtypes(include="number").columns.tolist()
            team_pred_summary = match_players[numeric_cols].sum().to_frame().T
            st.subheader(f"Predicted Team Totals for Match {match_id}")
            st.dataframe(team_pred_summary)
        else:
            st.info("No upcoming matches for team prediction")
