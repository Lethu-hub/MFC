import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import plotly.express as px

from utils.data_loader import load_all_data

st.set_page_config(page_title="MFC Predictions", layout="wide")
st.title("⚡ MFC Predictions: Player & Match Outcomes")

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
# Sidebar: Select target variable
# -----------------------------
st.sidebar.header("Prediction Setup")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

target_col = st.sidebar.selectbox("Select Target Variable", numeric_cols)
feature_cols = st.sidebar.multiselect("Select Features (X)", [c for c in df.columns if c != target_col])

if not feature_cols:
    st.warning("Please select at least one feature.")
else:
    # -----------------------------
    # Prepare data
    # -----------------------------
    X = df[feature_cols]
    y = df[target_col]

    # Encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # -----------------------------
    # Train Model
    # -----------------------------
    if st.sidebar.button("Train & Predict"):
        # Choose model type based on target
        if y.dtype in ["int64", "float64"] and y.nunique() > 5:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_type = "Regression"
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_type = "Classification"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"Model Type: {model_type}")
        st.write("### Predictions vs Actuals")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(results.head(20))

        # -----------------------------
        # Metrics
        # -----------------------------
        if model_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R² Score: {r2:.2f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.2f}")
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

        # -----------------------------
        # Feature Importance
        # -----------------------------
        st.write("### Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": X_encoded.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(feat_imp.head(20))

        fig_imp = px.bar(feat_imp.head(20), x="Feature", y="Importance", title="Top Features")
        st.plotly_chart(fig_imp, use_container_width=True)

