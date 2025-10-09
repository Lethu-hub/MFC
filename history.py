import streamlit as st
from utils.data_loader import load_all_data
from utils.stats import central_tendency, measures_of_spread, categorical_counts
from utils.charts import univariate_chart

st.set_page_config(page_title="MFC History", layout="wide")
st.title("📊 MFC History: Data Exploration & Univariate Analysis")

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
tab1, tab2, tab3 = st.tabs(["Raw Data", "Summary Stats", "Univariate Analysis"])

# -----------------------------
# Tab 1: Raw Data
# -----------------------------
with tab1:
    st.dataframe(df_filtered)

# -----------------------------
# Tab 2: Summary Stats
# -----------------------------
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

# -----------------------------
# Tab 3: Univariate Analysis
# -----------------------------
with tab3:
    col_to_plot = st.selectbox("Select Column", df_filtered.columns)
    chart_type = st.selectbox("Chart Type", ["Histogram", "Pie", "Boxplot", "Violin", "Cumulative"])
    fig = univariate_chart(df_filtered, col_to_plot, chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Cumulative chart requires a numeric column.")

