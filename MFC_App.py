# models/event_predictor.py

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

class EventPredictor:
    def __init__(self):
        self.data_folder = "data"
        self.models_folder = "models/trained"
        os.makedirs(self.models_folder, exist_ok=True)

    def load_data(self):
        """Load all datasets and prepare merged dataframe."""
        matches = pd.read_csv(os.path.join(self.data_folder, "matches.csv"))
        players = pd.read_csv(os.path.join(self.data_folder, "players.csv"))
        events = pd.read_csv(os.path.join(self.data_folder, "match_events.csv"))

        # Merge for richer features
        df = events.merge(players, on="Player_ID", how="left")
        df = df.merge(matches, on="Match_ID", how="left")

        # Convert date
        df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")
        df["Month"] = df["Match_Date"].dt.month

        self.df = df
        return df

    def train_all(self):
        """Train a regression model for each unique Event_Type."""
        if not hasattr(self, "df"):
            raise ValueError("Data not loaded. Call load_data() first.")

        results = []
        event_types = self.df["Event_Type"].unique()

        for event in event_types:
            df_event = (
                self.df[self.df["Event_Type"] == event]
                .groupby(["Season", "Match_ID"])
                .size()
                .reset_index(name="Event_Count")
            )

            # Create simple time-based features
            df_event["Match_Number"] = df_event.groupby("Season").cumcount() + 1

            # Features & Target
            X = df_event[["Match_Number"]]
            y = df_event["Event_Count"]

            if len(X) < 3:
                print(f"Skipping {event} (not enough data)")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Save model
            model_path = os.path.join(self.models_folder, f"{event}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            results.append({"Event": event, "R2": r2, "MAE": mae})
            print(f"✅ Trained {event} model — R2: {r2:.2f}, MAE: {mae:.2f}")

        return pd.DataFrame(results)
