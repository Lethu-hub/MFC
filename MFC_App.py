import streamlit as st
import pandas as pd
from data_loader import load_all_data
from stats import central_tendency, measures_of_spread, categorical_counts
from charts import univariate_chart
from datetime import datetime
from analytics_dashboard import display_analytics
import plotly.express as px
from event_predictor import EventPredictor

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
    st.title("📊 MFC Stats")
    st.markdown("Explore player and match data so far — Data Exploration & Stats")

    # Load datasets
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
    match_filter = st.sidebar.multiselect(
        "Select Match ID", options=df['Match_ID'].unique(), default=df['Match_ID'].unique()
    )

    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter)) &
        (df['Match_ID'].isin(match_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Player Stats", "Team Stats"])

    # Tab1: Data Exploration
    with tab1:
        st.subheader("📊 Explore Data")
        st.dataframe(df_filtered, width='stretch')

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

    # Tab2: Player Stats
    with tab2:
        st.subheader("Player-Level Summary")
        if df_filtered.empty:
            st.info("No data available for selected filters.")
        else:
            player_summary = df_filtered.groupby("Player_Name").agg(
                Matches_Played=("Match_ID", "nunique")
            ).reset_index()

            event_types = df_filtered['Event_Type'].unique()
            for event in event_types:
                player_summary[event] = player_summary['Player_Name'].apply(
                    lambda p: df_filtered[(df_filtered['Player_Name'] == p) & (df_filtered['Event_Type'] == event)].shape[0]
                )

            st.dataframe(
                player_summary.sort_values(by="Matches_Played", ascending=False),
                width='stretch'
            )

    # Tab3: Team Stats
    with tab3:
        st.subheader("🏆 Team-Level Summary (MFC)")

        if df_filtered.empty:
            st.info("No data available for selected filters.")
        else:
            # Auto-select players based on selected matches (ignore player_filter)
            selected_matches = match_filter
            match_players = df[df['Match_ID'].isin(selected_matches)]['Player_Name'].unique()
    
            # Show these players in a disabled multiselect
            st.multiselect("Players in Selected Match(es)", match_players, default=match_players, disabled=True)
    
            # Filter df_filtered only to these match players
            df_team = df_filtered[df_filtered['Player_Name'].isin(match_players)]
    
            # Aggregate team-level stats
            team_summary = df_team.groupby("Match_ID").agg(
                Total_Events=("Event_Type", "count")
            ).reset_index()
    
            # Count events by type
            event_types = df_team['Event_Type'].unique()
            for event in event_types:
                team_summary[event] = team_summary['Match_ID'].apply(
                    lambda m: df_team[(df_team['Match_ID'] == m) & (df_team['Event_Type'] == event)].shape[0]
                )
    
            st.dataframe(
                team_summary.sort_values(by="Match_ID", ascending=True),
                width='stretch'
            )

# ==========================
# Performance Page
# ==========================
elif page == "Performance":
    st.title("📈 MFC Performance Dashboard")
    st.markdown("Visualize player and team performance trends across matches and seasons.")

    import plotly.express as px
    from data_loader import load_all_data
    from analytics_dashboard import display_analytics  # Prebuilt analytics visuals

    # -----------------------------
    # Load datasets
    # -----------------------------
    dfs = load_all_data()
    players_df = dfs["Players"]
    matches_df = dfs["Matches"]
    events_df = dfs["Match Events"]

    # Merge player names for analysis
    df = events_df.merge(players_df, on="Player_ID", how="left")

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.sidebar.header("Filters")
    season_filter = st.sidebar.multiselect(
        "Select Season", options=df['Season'].unique(), default=list(df['Season'].unique())
    )
    player_filter = st.sidebar.multiselect(
        "Select Player", options=players_df['Player_Name'].unique(), default=list(players_df['Player_Name'].unique())
    )
    event_filter = st.sidebar.multiselect(
        "Select Event Type", options=df['Event_Type'].unique(), default=list(df['Event_Type'].unique())
    )
    match_filter = st.sidebar.multiselect(
        "Select Match ID", options=df['Match_ID'].unique(), default=list(df['Match_ID'].unique())
    )

    # Apply filters
    df_filtered = df[
        (df['Season'].isin(season_filter)) &
        (df['Player_Name'].isin(player_filter)) &
        (df['Event_Type'].isin(event_filter)) &
        (df['Match_ID'].isin(match_filter))
    ]

    st.write(f"Filtered dataset: {df_filtered.shape[0]:,} rows")

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Player Performance", "Team Performance", "Analytics"])

    # ==============================
    # Tab 1: Player Performance
    # ==============================
    with tab1:
        st.subheader("🏃 Player Performance Overview")

        if df_filtered.empty:
            st.info("No data for selected filters.")
        else:
            st.markdown("### 📊 Event Distribution per Player")
            fig1 = px.histogram(df_filtered, x="Player_Name", color="Event_Type",
                                title="Events by Player", barmode="group")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### 📈 Performance Over Matches")
            fig2 = px.line(df_filtered, x="Match_ID", y="Minute", color="Player_Name",
                           title="Performance Timeline per Player", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

        # --- Custom Chart Builder ---
        st.markdown("---")
        st.markdown("### 🧩 Custom Player Analysis")

        x_var = st.selectbox("Select X-axis", options=df_filtered.columns)
        y_var = st.selectbox("Select Y-axis", options=[col for col in df_filtered.columns if col != x_var])
        hue_var = st.selectbox("Select Color Group (optional)", options=[None] + list(df_filtered.columns))
        chart_type = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Pie"], horizontal=True)

        if st.button("Generate Player Chart"):
            if chart_type == "Bar":
                fig = px.bar(df_filtered, x=x_var, y=y_var, color=hue_var, title=f"{y_var} vs {x_var}")
            elif chart_type == "Line":
                fig = px.line(df_filtered, x=x_var, y=y_var, color=hue_var, markers=True, title=f"{y_var} vs {x_var}")
            elif chart_type == "Scatter":
                fig = px.scatter(df_filtered, x=x_var, y=y_var, color=hue_var, title=f"{y_var} vs {x_var}")
            elif chart_type == "Pie" and hue_var:
                fig = px.pie(df_filtered, names=hue_var, title=f"Distribution by {hue_var}")
            else:
                st.warning("Pie chart requires a color group (category) selected.")
                st.stop()

            st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Tab 2: Team Performance
    # ==============================
    with tab2:
        st.subheader("⚽ Team Performance Trends")

        if df_filtered.empty:
            st.info("No data for selected filters.")
        else:
            st.markdown("### 📊 Event Breakdown per Match")
            fig3 = px.histogram(df_filtered, x="Match_ID", color="Event_Type",
                                title="Event Counts per Match", barmode="group")
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### 🔁 Event Trends per Season")
            fig4 = px.bar(df_filtered, x="Season", color="Event_Type",
                          title="Events by Season", barmode="group")
            st.plotly_chart(fig4, use_container_width=True)

        # --- Custom Chart Builder ---
        st.markdown("---")
        st.markdown("### 🧩 Custom Team Analysis")

        x_var_team = st.selectbox("Select X-axis", options=df_filtered.columns, key="team_x")
        y_var_team = st.selectbox("Select Y-axis", options=[col for col in df_filtered.columns if col != x_var_team], key="team_y")
        hue_var_team = st.selectbox("Select Color Group (optional)", options=[None] + list(df_filtered.columns), key="team_hue")
        chart_type_team = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Pie"], horizontal=True, key="team_chart_type")

        if st.button("Generate Team Chart"):
            if chart_type_team == "Bar":
                fig_team = px.bar(df_filtered, x=x_var_team, y=y_var_team, color=hue_var_team, title=f"{y_var_team} vs {x_var_team}")
            elif chart_type_team == "Line":
                fig_team = px.line(df_filtered, x=x_var_team, y=y_var_team, color=hue_var_team, markers=True, title=f"{y_var_team} vs {x_var_team}")
            elif chart_type_team == "Scatter":
                fig_team = px.scatter(df_filtered, x=x_var_team, y=y_var_team, color=hue_var_team, title=f"{y_var_team} vs {x_var_team}")
            elif chart_type_team == "Pie" and hue_var_team:
                fig_team = px.pie(df_filtered, names=hue_var_team, title=f"Distribution by {hue_var_team}")
            else:
                st.warning("Pie chart requires a color group (category) selected.")
                st.stop()

            st.plotly_chart(fig_team, use_container_width=True)

    # ==============================
    # Tab 3: Analytics (Pre-built insights)
    # ==============================
    with tab3:
        st.subheader("🧠 Advanced Analytics Dashboard")
        st.markdown("These visuals summarize deeper insights like player age impact, top performers, and event patterns across seasons.")
        display_analytics()

        num_cols = 3
        cols = st.columns(num_cols)
    
        for i, event in enumerate(event_types):
            # Load model
            model_file = f"{event.replace(' ', '_')}_weekly_model.pkl"
            if not os.path.exists(model_file):
                st.warning(f"No model found for {event}.")
                continue
            model = joblib.load(model_file)
        
            # Prepare historical + predicted data (same as before)
            df_event = events_df[events_df['Event_Type'] == event].copy()
            df_hist = (
                df_event.groupby(['Year', 'Week'])
                .size()
                .reset_index(name='Event_Count')
            )
            df_hist['TimeIndex'] = (df_hist['Year'] - df_hist['Year'].min()) * 52 + df_hist['Week']
        
            last_time = df_hist['TimeIndex'].max()
            future_time = [last_time + w for w in range(1, future_weeks + 1)]
            pred_counts = model.predict(pd.DataFrame({'TimeIndex': future_time}))
            pred_counts = [max(0, round(x)) for x in pred_counts]
        
            df_plot = pd.concat([
                df_hist[['TimeIndex', 'Event_Count']],
                pd.DataFrame({'TimeIndex': future_time, 'Event_Count': pred_counts})
            ], ignore_index=True)
            df_plot['Type'] = ['Historical']*len(df_hist) + ['Predicted']*future_weeks
        
            # -----------------------------
            # Mini chart
            # -----------------------------
            mini_fig = px.line(
                df_plot,
                x='TimeIndex',
                y='Event_Count',
                color='Type',
                height=200,  # smaller height for mini graph
                markers=True
            )
            mini_fig.update_layout(margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
        
            col = cols[i % num_cols]
            with col:
                st.plotly_chart(mini_fig, use_container_width=True)
                if st.button(f"Expand {event}", key=f"btn_{i}"):
                    # Full-size chart
                    full_fig = px.line(
                        df_plot,
                        x='TimeIndex',
                        y='Event_Count',
                        color='Type',
                        markers=True,
                        title=f"{event} Predictions"
                    )
                    st.plotly_chart(full_fig, use_container_width=True)
# ==========================
# PREDICTIONS PAGE
# ==========================
elif page == "📊 Predictions":
    st.title("📊 Automatic Event Predictions")
    st.markdown("Each chart shows predicted event counts for upcoming matches based on historical data.")

    # Identify available models
    model_files = [f for f in os.listdir(".") if f.endswith("_model.pkl")]
    if not model_files:
        st.warning("⚠️ No trained models found in the current folder.")
        st.stop()

    event_types = [os.path.splitext(f)[0].replace("_model", "").replace("_", " ") for f in model_files]

    num_cols = 3
    cols = st.columns(num_cols)

    for i, event in enumerate(event_types):
        model_file = f"{event.replace(' ', '_')}_model.pkl"
        try:
            model = joblib.load(model_file)
        except Exception:
            continue

        # Prepare data for plotting
        event_df = match_data[match_data["Event_Type"] == event].copy()
        event_df = (
            event_df.groupby("Match_Date")
            .size()
            .reset_index(name="Event_Count")
            .sort_values("Match_Date")
        )

        if len(event_df) < 2:
            continue

        # Predict next event value
        last_value = event_df["Event_Count"].iloc[-1]
        try:
            prediction = model.predict([[last_value]])[0]
        except Exception:
            prediction = last_value

        # Build chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=event_df["Match_Date"],
            y=event_df["Event_Count"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="royalblue")
        ))
        fig.add_trace(go.Scatter(
            x=[event_df["Match_Date"].iloc[-1] + pd.Timedelta(days=7)],
            y=[prediction],
            mode="markers+text",
            name="Prediction",
            text=[f"{prediction:.1f}"],
            textposition="top center",
            marker=dict(color="orange", size=10)
        ))
        fig.update_layout(
            title=f"{event} Prediction Trend",
            xaxis_title="Match Date",
            yaxis_title="Event Count",
            template="plotly_white",
            hovermode="x unified",
            showlegend=False
        )

        # Display mini chart expandable on click
        with cols[i % num_cols]:
            with st.expander(f"📊 {event}", expanded=False):
                st.plotly_chart(fig, use_container_width=True)
