import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EventPredictor:
    def __init__(self, data_path="match_events.csv", model_dir="."):
        self.data_path = data_path
        self.model_dir = model_dir
        self.events_df = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df['Season'] = df['Season'].astype(str)
        self.events_df = df
        return df

    def train_all(self):
        if self.events_df is None:
            self.load_data()

        features = (
            self.events_df.groupby(['Season', 'Event_Type'])
            .size()
            .reset_index(name='Event_Count')
        )

        event_types = features['Event_Type'].unique()
        for event in event_types:
            df_event = features[features['Event_Type'] == event].copy()
            df_event['Lag'] = df_event['Event_Count'].shift(1).fillna(0)

            if len(df_event) < 3:
                continue

            X = df_event[['Lag']]
            y = df_event['Event_Count']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression().fit(X_train, y_train)
            joblib.dump(model, f"{self.model_dir}/{event}_model.pkl")

    def predict(self, event_type: str, last_value: float = 5):
        model_path = os.path.join(self.model_dir, f"{event_type}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {event_type}. Train models first.")
        
        model = joblib.load(model_path)
        prediction = model.predict([[last_value]])[0]
        return prediction

    # --- NEW METHOD ---
    def list_models(self):
        """Return a list of event types based on the .pkl models present."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith("_model.pkl")]
        return [os.path.splitext(f)[0].replace("_model", "") for f in model_files]
