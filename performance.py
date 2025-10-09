import streamlit as st
from utils.data_loader import load_all_data
from utils.charts import bivariate_chart, correlation_heatmap, pairplot, stacked_bar, timeseries_chart

st.set_page_config(page_title="MFC Performance Analysis", layout="wide")
st.title("⚽ MFC Performance Analysis: Bivariate & Multivariate Insights")

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
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Bivariate Charts", "Multivariate Charts", "Time Series"])

# -----------------------------
# Tab 1: Bivariate Charts
# -----------------------------
with tab1:
    x_col = st.selectbox("X-axis Column", df_filtered.columns, key="biv_x")
    y_col = st.selectbox("Y-axis Column", df_filtered.columns, key="biv_y")
    color_col = st.selectbox("Color Column (Optional)", [None] + df_filtered.columns.tolist(), key="biv_color")
    chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Box", "Bar", "Bubble"], key="biv_type")
    
    fig = bivariate_chart(df_filtered, x_col, y_col, chart_type, color_col)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2: Multivariate Charts
# -----------------------------
with tab2:
    st.subheader("Correlation Heatmap")
    fig_heat = correlation_heatmap(df_filtered)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Scatter Matrix / Pairplot")
    fig_pair = pairplot(df_filtered)
    st.plotly_chart(fig_pair, use_container_width=True)

# -----------------------------
# Tab 3: Time Series Charts
# -----------------------------
with tab3:
    x_time = st.selectbox("X-axis (Time Column)", df_filtered.columns, key="ts_x")
    y_time = st.selectbox("Y-axis (Metric Column)", df_filtered.columns, key="ts_y")
    color_time = st.selectbox("Color Column (Optional)", [None] + df_filtered.columns.tolist(), key="ts_color")
    
    fig_ts = timeseries_chart(df_filtered, x_time, y_time, color_time)
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True)

