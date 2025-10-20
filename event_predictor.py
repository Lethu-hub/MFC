# models/event_predictor.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

class EventPredictor:
    def __init__(self, data_path="match_events.csv", model_dir="models/trained"):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.events_df = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df['Season'] = df['Season'].astype(str)
        self.events_df = df
        return df

    def train_all(self):
        """
        Loop through every event type (Assist, Foul, Shot On Target, etc.)
        Train a simple regression model per event
        """
        if self.events_df is None:
            self.load_data()

        features = (
            self.events_df.groupby(['Season', 'Event_Type'])
            .size()
            .reset_index(name='Event_Count')
        )

        event_types = features['Event_Type'].unique()
        print(f"Training models for: {event_types}")

        for event in event_types:
            df_event = features[features['Event_Type'] == event].copy()
            df_event['Lag'] = df_event['Event_Count'].shift(1).fillna(0)

            if len(df_event) < 3:
                print(f"Skipping {event}: not enough samples")
                continue

            X = df_event[['Lag']]
            y = df_event['Event_Count']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression().fit(X_train, y_train)
            mse = mean_squared_error(y_test, model.predict(X_test))
            print(f"{event} model trained — MSE: {mse:.2f}")

            joblib.dump(model, f"{self.model_dir}/{event.replace(' ', '_')}_model.pkl")

    def predict(self, event_type: str, last_value: float = 5):
        """
        Predict the next event count based on the last event count (Lag)
        """
        model_path = f"{self.model_dir}/{event_type.replace(' ', '_')}_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {event_type}. Train models first.")
        
        model = joblib.load(model_path)
        prediction = model.predict([[last_value]])[0]
        return prediction
