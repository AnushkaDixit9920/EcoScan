import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="EcoScan", page_icon="🌍")

MODEL_PATH = "artifacts/model.pkl"
TRAINING_CSV_PATH = "data/Cleaned_Carbon_Emission.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_training_data():
    df = pd.read_csv(TRAINING_CSV_PATH)
    return df


def build_preprocessor(df):
    X = df.drop(columns=["CarbonEmission"], errors="ignore")

    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("scaler", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    preprocessor.fit(X)

    return preprocessor, cat_cols, num_cols

model = load_model()
train_df = load_training_data()
preprocessor, CAT_COLS, NUM_COLS = build_preprocessor(train_df)


st.title("🌍 EcoScan — Carbon Footprint Estimator")

with st.form("input_form"):
    st.subheader("Enter your lifestyle details")

    inputs = {}

    for c in CAT_COLS:
        opts = sorted(train_df[c].dropna().unique().tolist())
        inputs[c] = st.selectbox(c, opts)

    for n in NUM_COLS:
        default = float(train_df[n].median())
        inputs[n] = st.number_input(n, value=default)

    submit = st.form_submit_button("Estimate Carbon Footprint")

if submit:
    try:
        input_df = pd.DataFrame([inputs])
        input_df[NUM_COLS] = input_df[NUM_COLS].apply(pd.to_numeric)

        X_transformed = preprocessor.transform(input_df)
        pred = model.predict(X_transformed)[0]

        st.success(f"Your estimated carbon footprint: **{pred:.2f} kg CO₂ / month**")

    except Exception as e:
        st.error("Error during prediction.")
        st.exception(e)







