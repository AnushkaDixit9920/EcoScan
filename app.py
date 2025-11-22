import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="EcoScan", page_icon="🌿", layout="centered")

# ---------------------------------------------------------
# ECO GREEN CUSTOM CSS
# ---------------------------------------------------------
GREEN_UI = """
<style>
    body {
        background-color: #0e1411;
    }

    .main {
        background-color: #0e1411 !important;
    }

    .eco-title {
        font-size: 40px;
        font-weight: 800;
        color: #B8FFCC;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .eco-sub {
        font-size: 18px;
        text-align: center;
        color: #8ee6a6;
        margin-bottom: 25px;
    }

    .eco-card {
        background: #1a281f;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0px 4px 14px rgba(0,255,120,0.15);
        border: 1px solid rgba(0,255,120,0.15);
        margin-bottom: 25px;
    }

    .stButton>button {
        background-color: #0fdd74;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 16px;
        border: none;
        box-shadow: 0px 2px 10px rgba(0,255,120,0.3);
        transition: 0.2s ease-in-out;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #10f784;
        box-shadow: 0px 4px 14px rgba(0,255,120,0.5);
    }

    .result-box {
        background: #002f1a;
        border-left: 8px solid #0fdc72;
        padding: 18px;
        border-radius: 12px;
        color: #b8ffcc;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
        font-weight: 600;
        box-shadow: 0px 4px 14px rgba(0,255,120,0.15);
    }
</style>
"""

st.markdown(GREEN_UI, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL + PREPROCESSORS
# ---------------------------------------------------------
MODEL_PATH = "artifacts/model.pkl"
TRAINING_CSV_PATH = "data/Cleaned_Carbon_Emission.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_training_data():
    return pd.read_csv(TRAINING_CSV_PATH)

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

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<div class='eco-title'>🌿 EcoScan — Carbon Footprint Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='eco-sub'>Enter your lifestyle details to estimate your monthly carbon footprint</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT CARD
# ---------------------------------------------------------
st.markdown("<div class='eco-card'>", unsafe_allow_html=True)

with st.form("input_form"):
    st.subheader("🌱 Lifestyle Information")

    col1, col2 = st.columns(2)
    inputs = {}

    # Categorical fields split across two columns
    for i, c in enumerate(CAT_COLS):
        opts = sorted(train_df[c].dropna().unique().tolist())
        if i % 2 == 0:
            inputs[c] = col1.selectbox(f"🍃 {c}", opts)
        else:
            inputs[c] = col2.selectbox(f"🌿 {c}", opts)

    # Numeric fields split in two columns as well
    for i, n in enumerate(NUM_COLS):
        default_val = float(train_df[n].median())
        if i % 2 == 0:
            inputs[n] = col1.number_input(f"🔢 {n}", value=default_val)
        else:
            inputs[n] = col2.number_input(f"🔢 {n}", value=default_val)

    submit = st.form_submit_button("Estimate Carbon Footprint")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICTION HANDLING
# ---------------------------------------------------------
if submit:
    try:
        input_df = pd.DataFrame([inputs])
        input_df[NUM_COLS] = input_df[NUM_COLS].apply(pd.to_numeric)

        X_transformed = preprocessor.transform(input_df)
        pred = model.predict(X_transformed)[0]

        st.markdown(
            f"<div class='result-box'>🌎 Your estimated carbon footprint:<br><br><b>{pred:.2f} kg CO₂ / month</b></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)









