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
# FORCE GREEN BACKGROUND (REAL FIX)
# ---------------------------------------------------------
st.markdown("""
    <style>
        /* Force Streamlit into LIGHT mode */
        :root {
            color-scheme: light !important;
        }

        /* Main app background */
        .stApp {
            background-color: #8fffa0 !important;  /* vibrant eco green */
            background-image: none !important;
        }

        /* Remove white central container */
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1jicfl2, .st-emotion-cache-7oyrr6, .st-emotion-cache-1jtrq3p {
            background-color: #8fffa0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# WHITE CARD STYLING
# ---------------------------------------------------------
GREEN_CARDS = """
<style>

    .eco-title {
        font-size: 40px;
        font-weight: 700;
        color: #054c77;
        text-align: center;
        margin-top: 10px;
    }

    .eco-sub {
        font-size: 18px;
        text-align: center;
        color: #065a93;
        margin-bottom: 25px;
    }

    .eco-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,150,200,0.15);
        margin-bottom: 25px;
    }

    .stButton>button {
        background-color: #0fbf8a;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 16px;
        border: none;
        width: 100%;
        box-shadow: 0px 3px 14px rgba(0,150,120,0.3);
        transition: 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #16dca3;
        box-shadow: 0px 4px 18px rgba(0,150,120,0.5);
    }

    .result-box {
        background: #eafff3;
        border-left: 8px solid #0fbf8a;
        padding: 18px;
        border-radius: 12px;
        color: #06684e;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
        font-weight: 600;
        box-shadow: 0px 4px 18px rgba(0,150,120,0.15);
    }

</style>
"""

st.markdown(GREEN_CARDS, unsafe_allow_html=True)

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
st.markdown("<div class='eco-sub'>Make sustainable decisions with data-driven insights</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.markdown("<div class='eco-card'>", unsafe_allow_html=True)

with st.form("input_form"):
    st.subheader("🌱 Enter your lifestyle details")

    col1, col2 = st.columns(2)
    inputs = {}

    # Categorical fields
    for i, c in enumerate(CAT_COLS):
        options = sorted(train_df[c].dropna().unique().tolist())
        if i % 2 == 0:
            inputs[c] = col1.selectbox(f"🍃 {c}", options)
        else:
            inputs[c] = col2.selectbox(f"🌿 {c}", options)

    # Numeric fields
    for i, n in enumerate(NUM_COLS):
        default_val = float(train_df[n].median())
        if i % 2 == 0:
            inputs[n] = col1.number_input(f"🔢 {n}", value=default_val)
        else:
            inputs[n] = col2.number_input(f"🔢 {n}", value=default_val)

    submit = st.form_submit_button("Estimate Carbon Footprint")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICTION RESULT
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

