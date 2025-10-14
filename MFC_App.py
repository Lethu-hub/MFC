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

    # Apply filters
    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # -----------------------------
    # Team-level analytics
    # -----------------------------
    st.subheader("🏆 Team-Level Summary (MFC)")
    team_df = df_filtered[df_filtered['Club'].str.upper() == "MFC"]
    if not team_df.empty:
        team_summary = team_df.groupby('Event_Type')['Event_Value'].sum().reset_index()
        st.dataframe(team_summary)
    else:
        st.info("No team events found for selected filters.")

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Summary Stats", "Univariate Analysis"])

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

# ==========================
# Predictions Page
# ==========================
elif page == "Predictions":
    st.title("⚡ MFC Player Predictions & Team Analytics")

    # -----------------------------
    # Load Data
    # -----------------------------
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

    # -----------------------------
    # Upcoming Matches
    # -----------------------------
    if upcoming_df.empty:
        st.info("✅ No upcoming matches available for prediction.")
    else:
        mfc_upcoming = upcoming_df[
            (upcoming_df["HomeTeam"].str.upper() == "MFC") |
            (upcoming_df["AwayTeam"].str.upper() == "MFC")
        ]

        if mfc_upcoming.empty:
            st.info("No upcoming MFC matches scheduled.")
        else:
            st.subheader("📅 Upcoming MFC Matches")
            st.dataframe(
                mfc_upcoming[["Date", "HomeTeam", "AwayTeam", "Competition", "Venue"]],
                hide_index=True,
                use_container_width=True
            )

    # -----------------------------
    # Player Predictions
    # -----------------------------
    st.subheader("⚡ Player-Level Predictions")
    df = events_df.merge(players_df, on="Player_ID", how="inner")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    target_col = st.sidebar.selectbox("🎯 Target Variable", numeric_cols)
    feature_cols = st.sidebar.multiselect(
        "🧩 Features for Prediction",
        [c for c in df.columns if c != target_col]
    )

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]
        X_encoded = pd.get_dummies(X, drop_first=True)

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        import plotly.express as px

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        if st.sidebar.button("🚀 Train & Predict"):
            if y.dtype in ["int64", "float64"] and y.nunique() > 5:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_type = "Regression"
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_type = "Classification"

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader(f"🧠 Model Type: {model_type}")
            st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head(20))

            # Player-level predictions
            mfc_players = players_df[players_df["Club"].str.upper() == "MFC"].copy()
            upcoming_features = pd.DataFrame()

            for col in feature_cols:
                if col in df.columns:
                    player_avg = df.groupby("Player_ID")[col].mean()
                    upcoming_features[col] = mfc_players["Player_ID"].map(player_avg)

            upcoming_features = pd.get_dummies(upcoming_features, drop_first=True)
            upcoming_features = upcoming_features.reindex(columns=X_encoded.columns, fill_value=0)

            predictions = model.predict(upcoming_features)
            mfc_players["Predicted_" + target_col] = predictions

            st.dataframe(
                mfc_players[["Player_Name", "Predicted_" + target_col]].sort_values(
                    by="Predicted_" + target_col, ascending=False
                ).reset_index(drop=True)
            )

            # -----------------------------
            # Team-level predictions
            # -----------------------------
            st.subheader("🏆 Team-Level Predictions")
            team_summary = mfc_players[numeric_cols].sum().reset_index()
            team_summary.columns = ["Metric", "Total"]
            st.dataframe(team_summary)

            # Feature Importance
            feat_imp = pd.DataFrame({
                "Feature": X_encoded.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.write("### 🔍 Top Features Driving Predictions")
            st.dataframe(feat_imp.head(15))
            st.plotly_chart(
                px.bar(feat_imp.head(15), x="Feature", y="Importance", title="Feature Importance (Top 15)"),
                use_container_width=True
            )
    else:
        st.info("Select at least one feature to enable predictions.")

