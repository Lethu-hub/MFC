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
    st.title("📊 MFC History: Data Exploration & Univariate Analysis")

    from data_loader import load_all_data
    from stats import central_tendency, measures_of_spread, categorical_counts
    from charts import univariate_chart

    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    df = events_df.merge(players_df, on="Player_ID")

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

    tab1, tab2, tab3 = st.tabs(["Bivariate Charts", "Multivariate Charts", "Time Series"])

    with tab1:
        x_col = st.selectbox("X-axis Column", df_filtered.columns, key="biv_x")
        y_col = st.selectbox("Y-axis Column", df_filtered.columns, key="biv_y")
        color_col = st.selectbox("Color Column (Optional)", [None] + df_filtered.columns.tolist(), key="biv_color")
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Box", "Bar", "Bubble"], key="biv_type")

        fig = bivariate_chart(df_filtered, x_col, y_col, chart_type, color_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap")
        fig_heat = correlation_heatmap(df_filtered)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Scatter Matrix / Pairplot")
        fig_pair = pairplot(df_filtered)
        st.plotly_chart(fig_pair, use_container_width=True)

    with tab3:
        x_time = st.selectbox("X-axis (Time Column)", df_filtered.columns, key="ts_x")
        y_time = st.selectbox("Y-axis (Metric Column)", df_filtered.columns, key="ts_y")
        color_time = st.selectbox("Color Column (Optional)", [None] + df_filtered.columns.tolist(), key="ts_color")

        fig_ts = timeseries_chart(df_filtered, x_time, y_time, color_time)
        if fig_ts:
            st.plotly_chart(fig_ts, use_container_width=True)

# ==========================
# Predictions Page
# ==========================
elif page == "Predictions":
    st.title("⚡ MFC Predictions: Player & Match Outcomes")

    from utils.data_loader import load_all_data
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
    import plotly.express as px

    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    df = events_df.merge(players_df, on="Player_ID")

    st.sidebar.header("Prediction Setup")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    target_col = st.sidebar.selectbox("Select Target Variable", numeric_cols)
    feature_cols = st.sidebar.multiselect("Select Features (X)", [c for c in df.columns if c != target_col])

    if not feature_cols:
        st.warning("Please select at least one feature.")
    else:
        X = df[feature_cols]
        y = df[target_col]

        X_encoded = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        if st.sidebar.button("Train & Predict"):
            if y.dtype in ["int64", "float64"] and y.nunique() > 5:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_type = "Regression"
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_type = "Classification"

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader(f"Model Type: {model_type}")
            st.write("### Predictions vs Actuals")
            results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.dataframe(results.head(20))

            if model_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.2f}")
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

            st.write("### Feature Importance")
            feat_imp = pd.DataFrame({
                "Feature": X_encoded.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.dataframe(feat_imp.head(20))

            fig_imp = px.bar(feat_imp.head(20), x="Feature", y="Importance", title="Top Features")
            st.plotly_chart(fig_imp, use_container_width=True)
