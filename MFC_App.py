# ==========================
# Home Page
# ==========================
if page == "Home":
    st.subheader("Upcoming Matches")
    
    try:
        upcoming_matches_df = pd.read_csv("upcoming_matches.csv")
        # Convert date column
        upcoming_matches_df['Match_Date'] = pd.to_datetime(upcoming_matches_df['Match_Date'], errors='coerce')
        # Filter future matches
        upcoming_matches_df = upcoming_matches_df[upcoming_matches_df['Match_Date'] >= pd.Timestamp.today()]
    except FileNotFoundError:
        st.error("upcoming_matches.csv not found!")
        upcoming_matches_df = pd.DataFrame()

    if upcoming_matches_df.empty:
        st.info("No upcoming matches scheduled yet!")
    else:
        # Sort by date
        upcoming_matches_df = upcoming_matches_df.sort_values(by='Match_Date').head(3)

        # Display cards
        for _, match in upcoming_matches_df.iterrows():
            st.markdown(
                f"""
                <div style="
                    border:2px solid #007BFF; 
                    padding:20px; 
                    border-radius:15px; 
                    margin-bottom:15px; 
                    background:linear-gradient(135deg, #e0f0ff, #ffffff);
                    box-shadow: 3px 3px 8px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin:0; color:#007BFF;">{match['HomeTeam']} vs {match['AwayTeam']}</h3>
                    <p style="margin:5px 0; font-size:14px;">
                        📅 Date: <b>{match['Match_Date'].strftime('%A, %d %B %Y')}</b><br>
                        🏟️ Venue: {match['Venue']}<br>
                        ⚽ Competition: {match['Competition']}<br>
                        ⏰ Kick-Off: {match.get('KickOffTime', 'TBD')}
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
