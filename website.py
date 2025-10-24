import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="Sleep Disorder Checker", page_icon="😴", layout="centered")

# --- Data and Model Loading ---

# Cache loader (Streamlit v1.18+), fallback for older
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache

@cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """Load a model from a file path."""
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)

@cache_resource(show_spinner=False)
def load_metadata():
    """Load feature encoders and target class names from JSON files."""
    encoders_path = Path("encoders.json")
    targets_path = Path("target_classes.json")

    if not encoders_path.exists() or not targets_path.exists():
        st.error(
            "Metadata files not found. Please run the training notebook to generate "
            "`encoders.json` and `target_classes.json`."
        )
        st.stop()

    enc_map = json.loads(encoders_path.read_text())
    target_classes = json.loads(targets_path.read_text())
    return enc_map, target_classes

# --- Prediction and Data Processing ---

def get_expected_features(model):
    """Attempt to get the feature names the model was trained on."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # Fallback for pipelines
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None

def predict_with_optional_proba(model, X: pd.DataFrame):
    """Make predictions and get confidence scores if available."""
    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            P = model.predict_proba(X)
            # Get the probability of the predicted class
            classes = list(getattr(model, "classes_", []))
            if classes:
                cls_to_idx = {c: i for i, c in enumerate(classes)}
                idx = np.array([cls_to_idx.get(c, 0) for c in y_pred])
                proba = P[np.arange(len(y_pred)), idx]
            else:
                proba = P.max(axis=1)
        except Exception:
            proba = None  # Fail silently if proba fails
    return y_pred, proba

def apply_category_mappings(df: pd.DataFrame, enc_map: dict) -> pd.DataFrame:
    """Map string categorical columns to their numeric codes using the loaded encoder map."""
    df = df.copy()
    for col, mapping in enc_map.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(mapping)
    return df

def decode_predictions(y, target_classes: list):
    """Convert numeric predictions back to their string labels."""
    if not target_classes:
        return [str(v) for v in np.ravel(y)]  # Fallback

    output = []
    for v in np.ravel(y):
        try:
            idx = int(v)
            if 0 <= idx < len(target_classes):
                output.append(target_classes[idx])
            else:
                output.append(str(v))  # Index out of bounds
        except (ValueError, TypeError):
            output.append(str(v))  # Not a valid integer
    return output

# --- UI Rendering ---

def build_feature_ui_config(enc_map: dict) -> dict:
    """Dynamically create UI configuration from loaded encoders."""
    # Start with fixed numeric sliders
    feature_ui = {
        "Age": {"kind": "slider", "min": 18, "max": 100, "step": 1},
        "Sleep Duration": {"kind": "slider", "min": 0.0, "max": 14.0, "step": 0.25},
        "Quality of Sleep": {"kind": "slider", "min": 1, "max": 10, "step": 1},
        "Physical Activity Level": {"kind": "slider", "min": 0, "max": 300, "step": 5},
        "Stress Level": {"kind": "slider", "min": 0, "max": 10, "step": 1},
        "Heart Rate": {"kind": "slider", "min": 40, "max": 200, "step": 1},
        "Daily Steps": {"kind": "slider", "min": 1000, "max": 20000, "step": 100},
        "Systolic Blood Pressure": {"kind": "slider", "min": 90, "max": 180, "step": 1},
        "Diastolic Blood Pressure": {"kind": "slider", "min": 60, "max": 120, "step": 1},
    }
    # Add select boxes for categorical features
    for col, mapping in enc_map.items():
        feature_ui[col] = {"kind": "select", "options": list(mapping.keys())}
    return feature_ui

def render_feature_input(name: str, feature_ui_config: dict):
    """Render a Streamlit input widget based on the feature's configuration."""
    cfg = feature_ui_config.get(name)
    if not cfg:
        return st.text_input(name, key=name)  # Fallback for unknown features

    kind = cfg["kind"]
    if kind == "slider":
        min_v, max_v, step = cfg["min"], cfg["max"], cfg["step"]
        # Use float slider if any parameter is a float
        if any(isinstance(x, float) for x in (min_v, max_v, step)):
            return st.slider(name, float(min_v), float(max_v), step=float(step), key=name)
        return st.slider(name, int(min_v), int(max_v), step=int(step), key=name)

    if kind == "select":
        return st.selectbox(name, options=cfg["options"], key=name)

    return st.text_input(name, key=name) # Fallback

# --- Main App ---

st.title("Sleep Disorder Checker")
st.caption("An interactive interface for the sleep disorder prediction model.")

# Load model and metadata
model_path = Path("sleep_disorder_model.pkl")
if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`. Please train and save the model.")
    st.stop()

model = load_model(model_path)
ENC_MAP, TARGET_CLASSES = load_metadata()
FEATURE_UI = build_feature_ui_config(ENC_MAP)

expected_cols = get_expected_features(model)
if expected_cols:
    st.success(f"Model expects the following features: {', '.join(expected_cols)}")
else:
    st.warning("Could not automatically determine feature names from the model.")
    expected_cols = list(FEATURE_UI.keys()) # Fallback to UI config

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

# Single Prediction Tab
with tab_single:
    st.header("Predict a Single Case")
    with st.form("single_form"):
        inputs = {col: render_feature_input(col, FEATURE_UI) for col in expected_cols}
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Create a DataFrame from the inputs
            X_raw = pd.DataFrame([inputs], columns=expected_cols)

            # Convert categorical strings to numeric codes
            X_processed = apply_category_mappings(X_raw, ENC_MAP)

            # Make prediction
            y_pred, proba = predict_with_optional_proba(model, X_processed)

            # Decode numeric prediction to text label
            y_text = decode_predictions(y_pred, TARGET_CLASSES)

            st.success(f"**Prediction:** {y_text[0]}")
            if proba is not None:
                st.info(f"**Confidence:** {float(proba[0]):.2%}")

            with st.expander("View Input Data"):
                st.dataframe(X_processed)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Batch Prediction Tab
with tab_batch:
    st.header("Predict from a CSV File")
    st.caption("Upload a CSV file with the same columns the model was trained on.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded data preview ({df.shape[0]} rows):")
            st.dataframe(df.head())

            # Ensure all expected columns are present
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                st.error(f"The uploaded CSV is missing required columns: {missing_cols}")
                st.stop()

            # Prepare data for prediction
            df_in = df[expected_cols].copy()
            df_processed = apply_category_mappings(df_in, ENC_MAP)

            # Make predictions
            y_pred, proba = predict_with_optional_proba(model, df_processed)
            y_text = decode_predictions(y_pred, TARGET_CLASSES)

            # Add results to the output DataFrame
            out_df = df.copy()
            out_df["prediction_label"] = y_text
            if proba is not None:
                out_df["prediction_confidence"] = proba

            st.write("Prediction Results:")
            st.dataframe(out_df.head(20))

            # Provide download link for the results
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")

st.markdown("---")
st.caption("To run this app: `streamlit run website.py`")
