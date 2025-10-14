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
# History Page
# ==========================
elif page == "History":
    st.title("📊 MFC History")
    st.markdown("Explore player and match data so far — Data Exploration & Univariate Analysis")

    from data_loader import load_all_data
    from stats import central_tendency, measures_of_spread, categorical_counts
    from charts import univariate_chart

    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    df = events_df.merge(players_df, on="Player_ID")

    # ----------------------------- Sidebar filters
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

    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # ----------------------------- Tabs
    tab1, tab2, tab3, tab_team = st.tabs(["Raw Data", "Summary Stats", "Univariate Analysis", "🏆 Team Summary"])

    # Tab 1: Raw Data
    with tab1:
        st.dataframe(df_filtered)

    # Tab 2: Summary Stats
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

    # Tab 3: Univariate Analysis
    with tab3:
        col_to_plot = st.selectbox("Select Column", df_filtered.columns)
        chart_type = st.selectbox("Chart Type", ["Histogram", "Pie", "Boxplot", "Violin", "Cumulative"])
        fig = univariate_chart(df_filtered, col_to_plot, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cumulative chart requires a numeric column.")

    # Tab 4: Team Summary
    with tab_team:
        if not df_filtered.empty:
            match_id = st.selectbox("Select Match for Team Summary", df_filtered['Match_ID'].unique(), key="team_summary_history")
            match_players = df_filtered[df_filtered['Match_ID'] == match_id]
            numeric_cols = match_players.select_dtypes(include='number').columns.tolist()

            if numeric_cols:
                team_summary = match_players[numeric_cols].sum().to_frame().T
                st.subheader(f"Team Totals for Match {match_id}")
                st.dataframe(team_summary)
            else:
                st.info("No numeric stats available for this match.")
        else:
            st.info("No data available for team summary.")

# ==========================
# Performance Page
# ==========================
elif page == "Performance":
    st.title("⚽ MFC Performance Analysis: Bivariate & Multivariate Insights")

    from data_loader import load_all_data
    from charts import bivariate_chart, correlation_heatmap, pairplot, stacked_bar, timeseries_chart

    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

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

    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # Tabs
    tab1, tab2, tab3, tab_team = st.tabs(["Bivariate Charts", "Multivariate Charts", "Time Series", "🏆 Team Summary"])

    # Tab1: Bivariate Charts
    with tab1:
        x_col = st.selectbox("X-axis Column", df_filtered.columns, key="biv_x")
        y_col = st.selectbox("Y-axis Column", df_filtered.columns, key="biv_y")
        color_col = st.selectbox("Color Column (Optional)", [None]+df_filtered.columns.tolist(), key="biv_color")
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Box", "Bar", "Bubble"], key="biv_type")

        fig = bivariate_chart(df_filtered, x_col, y_col, chart_type, color_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Tab2: Correlation & Pairplot
    with tab2:
        st.subheader("Correlation Heatmap")
        fig_heat = correlation_heatmap(df_filtered)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Scatter Matrix / Pairplot")
        fig_pair = pairplot(df_filtered)
        st.plotly_chart(fig_pair, use_container_width=True)

    # Tab3: Time Series
    with tab3:
        x_time = st.selectbox("X-axis (Time Column)", df_filtered.columns, key="ts_x")
        y_time = st.selectbox("Y-axis (Metric Column)", df_filtered.columns, key="ts_y")
        color_time = st.selectbox("Color Column (Optional)", [None]+df_filtered.columns.tolist(), key="ts_color")

        fig_ts = timeseries_chart(df_filtered, x_time, y_time, color_time)
        if fig_ts:
            st.plotly_chart(fig_ts, use_container_width=True)

    # Tab4: Team Summary
    with tab_team:
        if not df_filtered.empty:
            match_id = st.selectbox("Select Match for Team Summary", df_filtered['Match_ID'].unique(), key="team_summary_perf")
            match_players = df_filtered[df_filtered['Match_ID'] == match_id]
            numeric_cols = match_players.select_dtypes(include='number').columns.tolist()

            if numeric_cols:
                team_summary = match_players[numeric_cols].sum().to_frame().T
                st.subheader(f"Team Totals for Match {match_id}")
                st.dataframe(team_summary)
            else:
                st.info("No numeric stats available for this match.")
        else:
            st.info("No data available for team summary.")

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
