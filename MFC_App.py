import streamlit as st
import pandas as pd

# ==========================
# Page config
# ==========================
st.set_page_config(
    page_title="MFC Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "History", "Performance", "Predictions"])

# ==========================
# Home Page (unchanged)
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
# Team Analytics Function
# ==========================
def team_event_summary(events_df, matches_df, team_name="MFC"):
    """Aggregate player events into team-level stats per match."""
    df = events_df.merge(matches_df, on="Match_ID", how="left")
    df_team = df[(df['HomeTeam'].str.upper() == team_name.upper()) |
                 (df['AwayTeam'].str.upper() == team_name.upper())]
    
    numeric_cols = df_team.select_dtypes(include="number").columns.tolist()
    team_summary = df_team.groupby('Match_ID')[numeric_cols].sum().reset_index()
    
    # Add match info back
    team_summary = team_summary.merge(
        matches_df[['Match_ID', 'Date', 'HomeTeam', 'AwayTeam', 'Competition']],
        on='Match_ID', how='left'
    )
    return team_summary

# ==========================
# History Page
# ==========================
elif page == "History":
    st.title("📊 MFC History")
    st.markdown("Explore player and match data so far — Data Exploration & Univariate Analysis")

    from data_loader import load_all_data
    from stats import central_tendency, measures_of_spread, categorical_counts
    from charts import univariate_chart

    # Load datasets
    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    # Merge player names
    df = events_df.merge(players_df, on="Player_ID")

    # Sidebar filters
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

    # Apply filters
    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]
    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # -----------------------------
    # Show team-level summary
    team_summary = team_event_summary(events_df, matches_df, team_name="MFC")
    st.subheader("⚽ MFC Team-Level Stats")
    st.dataframe(team_summary, use_container_width=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Summary Stats", "Univariate Analysis"])
    with tab1:
        st.dataframe(df_filtered)

    with tab2:
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

    with tab3:
        col_to_plot = st.selectbox("Select Column", df_filtered.columns)
        chart_type = st.selectbox("Chart Type", ["Histogram", "Pie", "Boxplot", "Violin", "Cumulative"])
        fig = univariate_chart(df_filtered, col_to_plot, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cumulative chart requires a numeric column.")

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
            st.error("⚠️ Missing one or more core datasets (players.csv, match_events.csv).")
            return None, None, pd.DataFrame()

        try:
            upcoming_df = pd.read_csv("upcoming_matches.csv", parse_dates=["Date"])
        except FileNotFoundError:
            st.warning("⚠️ 'upcoming_matches.csv' not found. No future matches available.")
            upcoming_df = pd.DataFrame()

        return players_df, events_df, upcoming_df

    players_df, events_df, upcoming_df = load_data()
    if players_df is None or events_df is None:
        st.stop()

    # Show team-level stats
    team_summary = team_event_summary(events_df, pd.read_csv("matches.csv"), team_name="MFC")
    st.subheader("⚽ MFC Team-Level Stats")
    st.dataframe(team_summary, use_container_width=True)

    # Remaining player predictions code here (unchanged)
    # ...
