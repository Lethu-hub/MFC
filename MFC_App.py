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
# ==========================
# Statistics Page
# ==========================
elif page == "Statistics":
    st.title("📊 MFC History & Stats")
    st.markdown("Explore player and match data so far — Data Exploration & Stats")

    from data_loader import load_all_data
    from stats import central_tendency, measures_of_spread, categorical_counts

    # -----------------------------
    # Load datasets
    # -----------------------------
    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    df = events_df.merge(players_df, on="Player_ID", how="left")

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.sidebar.header("Filters")
    season_filter = st.sidebar.multiselect(
        "Select Season", options=df['Season'].unique(), default=df['Season'].unique()
    )
    player_filter = st.sidebar.multiselect(
        "Select Player", options=df['Player_Name'].unique(), default=df['Player_Name'].unique()
    )
    event_filter = st.sidebar.multiselect(
        "Select Event Type", options=df['Event_Type'].unique(), default=df['Event_Type'].unique()
    )
    match_filter = st.sidebar.selectbox(
        "Select Match (optional)", options=[None] + df['Match_ID'].unique().tolist()
    )

    # Apply filters
    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]
    if match_filter:
        df_filtered = df_filtered[df_filtered['Match_ID'] == match_filter]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # -----------------------------
    # Tabs: Data Exploration / Player Stats / Team Stats
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Player Stats", "Team Stats"])

    # -----------------------------
    # Tab 1: Data Exploration
    # -----------------------------
    with tab1:
        st.subheader("Raw Data")
        st.dataframe(df_filtered, use_container_width=True)

        numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
        cat_cols = df_filtered.select_dtypes(include='object').columns.tolist()

        if numeric_cols:
            st.subheader("Central Tendency")
            st.write(central_tendency(df_filtered, numeric_cols))
            st.subheader("Measures of Spread")
            st.write(measures_of_spread(df_filtered, numeric_cols))

        if cat_cols:
            st.subheader("Categorical Counts")
            counts = categorical_counts(df_filtered, cat_cols)
            for col, series in counts.items():
                st.write(f"Counts for {col}")
                st.write(series)

    # -----------------------------
    # Tab 2: Player Stats
    # -----------------------------
    with tab2:
        st.subheader("Player-Level Summary")
        if df_filtered.empty:
            st.info("No data available for selected filters.")
        else:
            agg_dict = {
                "Match_ID": pd.Series.nunique
            }
            for event in df_filtered['Event_Type'].unique():
                agg_dict[event] = lambda x, e=event: (x == e).sum()

            player_summary = df_filtered.groupby("Player_Name").agg(**agg_dict).reset_index()
            player_summary.rename(columns={"Match_ID": "Matches_Played"}, inplace=True)
            st.dataframe(player_summary.sort_values(by="Matches_Played", ascending=False), use_container_width=True)

    # -----------------------------
    # Tab 3: Team Stats
    # -----------------------------
    with tab3:
        st.subheader("Team-Level Summary")
        if df_filtered.empty:
            st.info("No data available for selected filters.")
        else:
            agg_dict = {}
            for event in df_filtered['Event_Type'].unique():
                agg_dict[event] = lambda x, e=event: (x == e).sum()
            team_summary = df_filtered.agg(**agg_dict).to_frame(name="Total").reset_index().rename(columns={"index": "Event_Type"})
            st.dataframe(team_summary, use_container_width=True)

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
