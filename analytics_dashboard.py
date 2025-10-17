import pandas as pd
import streamlit as st
import plotly.express as px

# ============================
# 📊 MAIN FUNCTION
# ============================
def display_analytics():
    st.title("📊 Player & Match Analytics Dashboard")

    # --- Load data ---
    players = pd.read_csv("players.csv")
    matches = pd.read_csv("matches.csv")
    events = pd.read_csv("match_events.csv")

    # --- Merge data ---
    df = events.merge(players, on="Player_ID", how="left")
    df = df.merge(matches, on=["Match_ID", "Season", "Match_Date"], how="left")

    # --- Convert Match_Date ---
    df["Match_Date"] = pd.to_datetime(df["Match_Date"])
    df["Month"] = df["Match_Date"].dt.month
    df["Month_Name"] = df["Match_Date"].dt.strftime("%B")

    # --- Map seasons (Winter, Spring, etc.) ---
    def season_label(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    df["Season_Type"] = df["Month"].apply(season_label)

    # --- Define charts section ---
    st.header("🏅 Top Player Performance")

    # 1️⃣ Top 10 Players by Goals
    goals = df[df["Event_Type"] == "Goal"].groupby("Player_Name").size().reset_index(name="Goals")
    fig1 = px.bar(goals.sort_values("Goals", ascending=False).head(10),
                  x="Player_Name", y="Goals", title="Top 10 Players by Goals", color="Goals")
    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Top 10 Players by Assists
    assists = df[df["Event_Type"] == "Assist"].groupby("Player_Name").size().reset_index(name="Assists")
    fig2 = px.bar(assists.sort_values("Assists", ascending=False).head(10),
                  x="Player_Name", y="Assists", title="Top 10 Players by Assists", color="Assists")
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Top 10 Players by Shots on Target
    shots_on = df[df["Event_Type"] == "Shot On Target"].groupby("Player_Name").size().reset_index(name="Shots_On_Target")
    fig3 = px.bar(shots_on.sort_values("Shots_On_Target", ascending=False).head(10),
                  x="Player_Name", y="Shots_On_Target", title="Top 10 Players by Shots On Target", color="Shots_On_Target")
    st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Total Shots (On + Off)
    total_shots = df[df["Event_Type"].isin(["Shot On Target", "Shot Off Target"])]
    total_shots = total_shots.groupby("Player_Name").size().reset_index(name="Total_Shots")
    fig4 = px.bar(total_shots.sort_values("Total_Shots", ascending=False).head(10),
                  x="Player_Name", y="Total_Shots", title="Top 10 Players by Total Shots", color="Total_Shots")
    st.plotly_chart(fig4, use_container_width=True)

    # 5️⃣ Age vs Goals Scored
    goals_age = df[df["Event_Type"] == "Goal"].groupby("Age").size().reset_index(name="Goals")
    fig5 = px.line(goals_age, x="Age", y="Goals", title="Goals Scored by Player Age", markers=True)
    st.plotly_chart(fig5, use_container_width=True)

    # 6️⃣ Event Distribution
    event_counts = df["Event_Type"].value_counts().reset_index()
    event_counts.columns = ["Event_Type", "Count"]
    fig6 = px.pie(event_counts, names="Event_Type", values="Count", title="Distribution of Event Types")
    st.plotly_chart(fig6, use_container_width=True)

    # 7️⃣ Performance by Season
    season_perf = df[df["Event_Type"].isin(["Goal", "Assist"])].groupby(["Season", "Event_Type"]).size().reset_index(name="Count")
    fig7 = px.bar(season_perf, x="Season", y="Count", color="Event_Type", barmode="group", title="Performance by Season")
    st.plotly_chart(fig7, use_container_width=True)

    # 8️⃣ Performance by Season Type (Winter, Summer, etc.)
    season_type_perf = df[df["Event_Type"].isin(["Goal", "Assist"])].groupby(["Season_Type", "Event_Type"]).size().reset_index(name="Count")
    fig8 = px.bar(season_type_perf, x="Season_Type", y="Count", color="Event_Type", barmode="group", title="Performance by Seasonal Period")
    st.plotly_chart(fig8, use_container_width=True)

    # 9️⃣ Events per Player Position
    position_perf = df.groupby(["Position", "Event_Type"]).size().reset_index(name="Count")
    fig9 = px.bar(position_perf, x="Position", y="Count", color="Event_Type", title="Average Events per Player Position")
    st.plotly_chart(fig9, use_container_width=True)

    # 🔟 Event Trends Over Time
    time_trend = df.groupby(df["Match_Date"].dt.to_period("M")).size().reset_index(name="Event_Count")
    time_trend["Match_Date"] = time_trend["Match_Date"].astype(str)
    fig10 = px.line(time_trend, x="Match_Date", y="Event_Count", title="Event Trend Over Time", markers=True)
    st.plotly_chart(fig10, use_container_width=True)

    # --- Summary stats ---
    st.header("📈 Summary Stats")
    total_goals = goals["Goals"].sum() if not goals.empty else 0
    total_assists = assists["Assists"].sum() if not assists.empty else 0
    total_shots_count = total_shots["Total_Shots"].sum() if not total_shots.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Goals", total_goals)
    col2.metric("Total Assists", total_assists)
    col3.metric("Total Shots", total_shots_count)

    st.success("✅ Analytics Dashboard loaded successfully!")


# ============================
# 📦 Run in isolation
# ============================
if __name__ == "__main__":
    display_analytics()
