# forms.py
import streamlit as st
import pandas as pd
from datetime import date

# =========================
# PLAYER FORM & MANAGEMENT
# =========================
def player_form(supabase):
    st.subheader("👥 Add New Player")
    with st.form("add_player_form", clear_on_submit=True):
        first_name = st.text_input("First Name *")
        surname = st.text_input("Surname *")
        date_of_birth = st.date_input("Date of Birth *", value=date(2000, 1, 1))
        nationality = st.text_input("Nationality *")
        position = st.selectbox("Position *", ["", "Goalkeeper", "Defender", "Midfielder", "Forward"])
        jersey_number = st.number_input("Jersey Number *", min_value=1, step=1)
        height_cm = st.number_input("Height (cm)", min_value=0)
        weight_kg = st.number_input("Weight (kg)", min_value=0)
        submit_player = st.form_submit_button("➕ Add Player")

        if submit_player:
            if not (first_name and surname and nationality and position):
                st.error("⚠️ Please fill in all required fields marked with *")
            else:
                data = {
                    "first_name": first_name,
                    "surname": surname,
                    "date_of_birth": str(date_of_birth),
                    "nationality": nationality,
                    "position": position,
                    "jersey_number": int(jersey_number),
                    "height_cm": int(height_cm),
                    "weight_kg": int(weight_kg)
                }
                response = supabase.table("players").insert(data).execute()
                if response.data:
                    st.success(f"✅ Player '{first_name} {surname}' added successfully!")
                else:
                    st.error("❌ Failed to add player")

    st.divider()
    st.subheader("📋 Manage Players")
    players = supabase.table("players").select("*").execute().data
    if players:
        df_players = pd.DataFrame(players)
        st.dataframe(df_players, use_container_width=True)

        # Delete using a selectbox instead of manual ID input
        delete_id = st.selectbox("Select Player to Delete", options=[p['player_id'] for p in players])
        if st.button("🗑️ Delete Player"):
            supabase.table("players").delete().eq("player_id", delete_id).execute()
            st.success("✅ Player deleted successfully! Refresh to update list.")
    else:
        st.info("No players found.")


# =========================
# MATCH FORM & MANAGEMENT
# =========================
def match_form(supabase):
    st.subheader("🏆 Add New Match")
    with st.form("add_match_form", clear_on_submit=True):
        match_date = st.date_input("Match Date *", value=date.today())
        opponent = st.text_input("Opponent *")
        venue = st.text_input("Venue")
        result = st.selectbox("Result", ["", "Win", "Loss", "Draw"])
        score_mfc = st.number_input("MFC Score", min_value=0, step=1)
        score_opponent = st.number_input("Opponent Score", min_value=0, step=1)
        season = st.text_input("Season (e.g., 2024/2025)")
        submit_match = st.form_submit_button("➕ Add Match")

        if submit_match:
            if not (match_date and opponent):
                st.error("⚠️ Please fill in all required fields marked with *")
            else:
                data = {
                    "match_date": str(match_date),
                    "opponent": opponent,
                    "venue": venue,
                    "result": result,
                    "score_mfc": int(score_mfc),
                    "score_opponent": int(score_opponent),
                    "season": season
                }
                response = supabase.table("matches").insert(data).execute()
                if response.data:
                    st.success(f"✅ Match vs '{opponent}' added successfully!")
                else:
                    st.error("❌ Failed to add match")

    st.divider()
    st.subheader("📋 Manage Matches")
    matches = supabase.table("matches").select("*").execute().data
    if matches:
        df_matches = pd.DataFrame(matches)
        st.dataframe(df_matches, use_container_width=True)

        delete_id = st.selectbox("Select Match to Delete", options=[m['match_id'] for m in matches])
        if st.button("🗑️ Delete Match"):
            supabase.table("matches").delete().eq("match_id", delete_id).execute()
            st.success("✅ Match deleted successfully! Refresh to update list.")
    else:
        st.info("No matches found.")


# =========================
# MATCH EVENTS FORM & MANAGEMENT
# =========================
def match_event_form(supabase):
    st.subheader("🎯 Add New Match Event")
    players = supabase.table("players").select("*").execute().data
    matches = supabase.table("matches").select("*").execute().data

    with st.form("add_event_form", clear_on_submit=True):
        match_id = st.selectbox("Match *", options=[f"{m['match_id']} vs {m['opponent']}" for m in matches])
        player_id = st.selectbox("Player *", options=[f"{p['player_id']} - {p['first_name']} {p['surname']}" for p in players])
        event_type = st.selectbox("Event Type *", ["Goal", "Assist", "Foul", "Substitution", "Injury", "Card", "Other"])
        minute = st.number_input("Minute", min_value=0, step=1)
        description = st.text_area("Description")
        season = st.text_input("Season")
        submit_event = st.form_submit_button("➕ Add Event")

        if submit_event:
            # Extract IDs from selectbox strings
            match_id_val = int(match_id.split()[0])
            player_id_val = int(player_id.split()[0])

            data = {
                "match_id": match_id_val,
                "player_id": player_id_val,
                "event_type": event_type,
                "minute": int(minute),
                "description": description,
                "season": season
            }
            response = supabase.table("match_events").insert(data).execute()
            if response.data:
                st.success("✅ Match event added successfully!")
            else:
                st.error("❌ Failed to add event")

    st.divider()
    st.subheader("📋 Manage Match Events")
    events = supabase.table("match_events").select("*").execute().data
    if events:
        df_events = pd.DataFrame(events)
        st.dataframe(df_events, use_container_width=True)

        delete_id = st.selectbox("Select Event to Delete", options=[e['match_event_id'] for e in events])
        if st.button("🗑️ Delete Event"):
            supabase.table("match_events").delete().eq("match_event_id", delete_id).execute()
            st.success("✅ Event deleted successfully! Refresh to update list.")
    else:
        st.info("No match events found.")
