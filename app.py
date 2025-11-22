

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer

st.markdown('---')

st.set_page_config(layout="wide", page_title="Loan Approval System")
st.title("Loan Approval System — EDA + Prediction")

# -------------------------
# Helpers
# # -------------------------
def nice_label(colname: str) -> str:
    """Pretty-print column name for UI (simple heuristic)."""
    if colname is None:
        return ""
    s = str(colname)
    s = s.replace('_', ' ').replace('.', ' ').strip()
    return s.title()

def contains_digit(s: str) -> bool:
    """Return True if string contains any digit 0-9."""
    if s is None:
        return False
    return any(ch.isdigit() for ch in str(s))

def is_effectively_numeric(val) -> bool:
    """Checks if a value is a numeric type or can be cast to float (safe guard)."""
    try:
        float(val)
        return True
    except Exception:
        return False

def infer_feature_names(model_obj):
    """Best-effort extraction of feature names from a trained model/pipe."""
    try:
        if hasattr(model_obj, 'feature_names_in_'):
            return list(model_obj.feature_names_in_)
        if hasattr(model_obj, 'get_feature_names_out'):
            return list(model_obj.get_feature_names_out())
    except Exception:
        pass
    try:
        if hasattr(model_obj, 'named_steps'):
            for step in model_obj.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
    except Exception:
        pass
    return None

# -------------------------
# Load dataset + model silently (cached)
# -------------------------
@st.cache_data
def load_dataset(path="loan_approval.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="loan_approval.joblib"):
    return joblib.load(path)

# Attempt to load dataset & model; show a single error if missing/unreadable
try:
    df = load_dataset()
except Exception as e:
    st.error("Could not find or read 'loan_approval.csv' in the app folder. Please place the CSV file next to this script.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("Could not find or load 'loan_approval.joblib' in the app folder. Please place the model file next to this script.")
    st.stop()

# -------------------------
# Show only dataset preview (head)
# -------------------------
st.subheader("Dataset preview")
st.dataframe(df.head())  # only first 5 rows shown

# -------------------------
# Simple interactive EDA (keeps functionality but no extra dataset stats)
# -------------------------
st.subheader("Exploratory Data Analysis (interactive)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

colA, colB = st.columns([2, 1])
with colA:
    plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Bar (categorical)", "Scatter", "Correlation heatmap"])
with colB:
    if plot_type in ("Histogram", "Boxplot"):
        feat = st.selectbox("Feature", numeric_cols, format_func=nice_label)
    elif plot_type == "Bar (categorical)":
        feat = st.selectbox("Categorical", cat_cols, format_func=nice_label)
    elif plot_type == "Scatter":
        x_feat = st.selectbox("X (numeric)", numeric_cols, index=0, format_func=nice_label)
        y_feat = st.selectbox("Y (numeric)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, format_func=nice_label)
        color = st.selectbox("Color by (optional)", ["None"] + cat_cols)
    else:
        corr_feats = st.multiselect("Correlation features (numeric)", numeric_cols, default=numeric_cols[:8], format_func=nice_label)

if st.button("Draw"):
    try:
        if plot_type == "Histogram":
            fig = px.histogram(df, x=feat, nbins=30, title=f"Histogram: {nice_label(feat)}")
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Boxplot":
            fig = px.box(df, y=feat, points='all', title=f"Boxplot: {nice_label(feat)}")
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Bar (categorical)":
            counts = df[feat].value_counts().reset_index()
            counts.columns = [feat, 'count']
            fig = px.bar(counts, x=feat, y='count', title=f"Counts: {nice_label(feat)}")
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Scatter":
            if x_feat == y_feat:
                st.error("Choose different features for X and Y")
            else:
                if color != "None":
                    fig = px.scatter(df, x=x_feat, y=y_feat, color=color, trendline='ols', title=f"{nice_label(y_feat)} vs {nice_label(x_feat)}")
                else:
                    fig = px.scatter(df, x=x_feat, y=y_feat, trendline='ols', title=f"{nice_label(y_feat)} vs {nice_label(x_feat)}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            use = corr_feats if corr_feats else numeric_cols
            if len(use) < 2:
                st.error("Select at least 2 numeric features for correlation")
            else:
                corr = df[use].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax)
                st.pyplot(fig)
    except Exception:
        st.error("Failed to draw plot")
        with st.expander("Traceback"):
            st.text(traceback.format_exc())

# -------------------------
# Model feature inference (best-effort)
# -------------------------
MODEL_FEATURES = infer_feature_names(model) if model is not None else None

# -------------------------
# Prediction interface (strict validation)
# -------------------------
st.subheader(" Prediction ")

# Suggest features to use for prediction
if MODEL_FEATURES:
    suggested = [f for f in MODEL_FEATURES if f not in ('Loan_Status',)]
else:
    suggested = [c for c in df.columns if c not in ('Loan_Status',)]

# Ensure 'name' and 'city' present near front if available
if 'name' in df.columns and 'name' not in suggested:
    suggested.insert(0, 'name')
if 'city' in df.columns and 'city' not in suggested:
    suggested.insert(0, 'city')

selected = st.multiselect("Select features for prediction (order should match model)", options=suggested, default=suggested)

# Define text-only features
text_only_features = {'name', 'city'}

if selected:
    with st.form('predict'):
        st.write("NOTE: 'name' and 'city' must be text (no digits). Other features are numeric-only.")
        inputs = {}
        # layout: two columns for inputs for nicer UI
        idx = 0
        col_left, col_right = None, None
        for f in selected:
            # create columns in pairs
            if idx % 2 == 0:
                col_left, col_right = st.columns(2)
            target_col = col_left if idx % 2 == 0 else col_right
            idx += 1

            # If feature is explicitly text-only (name/city) -> text_input
            if str(f).lower() in text_only_features:
                # default: most frequent non-null value from dataset if available
                default_text = ""
                if f in df.columns:
                    try:
                        default_text = str(df[f].dropna().mode().iloc[0])
                    except Exception:
                        default_text = ""
                inputs[f] = target_col.text_input(nice_label(f) + " (text only)", value=default_text, key=f"txt_{f}")

            else:
                # numeric-only input: prefer median if available
                default_val = 0.0
                if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                    series = df[f].dropna()
                    if not series.empty:
                        default_val = float(series.median())
                # Use number_input (which enforces numeric typing)
                inputs[f] = target_col.number_input(nice_label(f) + " (numeric only)", value=default_val, format="%.6f", key=f"num_{f}")

        submit = st.form_submit_button("Predict")

    if submit:
        # Validate text-only fields: ensure they contain at least one alphabetic char and no digits
        text_errors = []
        for f in selected:
            if str(f).lower() in text_only_features:
                v = inputs.get(f, "")
                # Reject if contains any digit
                if contains_digit(v):
                    text_errors.append(f)

                # Additionally ensure there's at least one alphabetic character (not entirely symbols)
                if not any(ch.isalpha() for ch in str(v)):
                    # treat empty or non-letter entries as error
                    if f not in text_errors:
                        text_errors.append(f)

        # Validate numeric fields: number_input already ensures numeric, but check again
        numeric_errors = []
        for f in selected:
            if str(f).lower() not in text_only_features:
                v = inputs.get(f, None)
                if v is None or (not is_effectively_numeric(v)):
                    numeric_errors.append(f)

        # If any errors, show helpful messages and skip prediction
        if text_errors or numeric_errors:
            if text_errors:
                st.error("Please fix these text fields (must be characters only, no digits):")
                for fe in text_errors:
                    st.warning(f"- {fe} (enter letters/words only, e.g. 'Ram Kumar' or 'Kathmandu')")
            if numeric_errors:
                st.error("Please fix these numeric fields (must be numbers only):")
                for fe in numeric_errors:
                    st.warning(f"- {fe} (enter a numeric value, e.g. 12345 or 2500.50)")
            st.stop()

        # Build DataFrame preserving selected order
        X = pd.DataFrame([{k: inputs[k] for k in selected}], columns=selected)

        st.write("Input values:")
        st.table(X.T.rename(columns={0: 'value'}))

        # Basic preprocessing: impute numeric columns if any
        try:
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) > 0:
                imp = SimpleImputer(strategy='median')
                X[num_cols] = imp.fit_transform(X[num_cols])

            # Force text-only columns to string dtype
            for col in X.columns:
                if col.lower() in text_only_features:
                    X[col] = X[col].astype(str)

            # Run prediction
            try:
                yhat = model.predict(X)
            except Exception:
                # Some models expect numpy array order
                yhat = model.predict(X.values)

            out = yhat[0] if hasattr(yhat, '__len__') else yhat

           
            if bool(out):  
                st.success("Congratulation! Your application is approved.")  # Green box
            else:
                st.error("Sorry, Your application is rejected.")   # Red box

         



            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    st.write("Prediction probabilities:")
                    st.write(proba)
                except Exception:
                    pass

        except Exception:
            st.error("Prediction failed. See traceback for details.")
            with st.expander("Traceback"):
                st.text(traceback.format_exc())

st.markdown('---')

