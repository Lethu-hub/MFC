# event_predictor.py
import os
import joblib
import pandas as pd

class EventPredictor:
    def __init__(self, data_path="match_events.csv", model_dir="."):
        """
        model_dir defaults to current directory (.)
        where .pkl files like Assist.pkl, Foul.pkl, etc. are stored.
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.events_df = None

    def load_data(self):
        """Load match events data if available."""
        if not os.path.exists(self.data_path):
            print(f"⚠️ Warning: {self.data_path} not found — continuing without it.")
            return None
        df = pd.read_csv(self.data_path)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df['Season'] = df['Season'].astype(str)
        self.events_df = df
        return df

    def list_models(self):
        """Find all .pkl model files in the current directory."""
        models = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]
        event_types = [m.replace(".pkl", "").replace("_", " ") for m in models]
        return event_types

    def predict_all(self, last_value: float = 5):
        """
        Load each .pkl model and generate a prediction.
        Returns a summary DataFrame.
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]

        if not model_files:
            raise FileNotFoundError("No .pkl models found in the current directory!")

        results = []
        for model_file in model_files:
            event_type = model_file.replace(".pkl", "").replace("_", " ")
            model_path = os.path.join(self.model_dir, model_file)

            try:
                model = joblib.load(model_path)
                prediction = model.predict([[last_value]])[0]
                results.append({
                    "Event_Type": event_type,
                    "Last_Value": last_value,
                    "Predicted_Value": round(prediction, 2)
                })
            except Exception as e:
                results.append({
                    "Event_Type": event_type,
                    "Error": str(e)
                })

        return pd.DataFrame(results)
