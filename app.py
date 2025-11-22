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
# FIX STREAMLIT THEME + APPLY MINT GREEN BACKGROUND
# ---------------------------------------------------------
st.markdown("""
    <style>
        :root {
            color-scheme: light !important;
        }

        .stApp {
            background-color: #c8f7cc !important;
            background-image: none !important;
        }

        .st-emotion-cache-18ni7ap,
        .st-emotion-cache-1jicfl2,
        .st-emotion-cache-7oyrr6,
        .st-emotion-cache-1jtrq3p {
            background-color: #c8f7cc !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# STYLING (TITLE + CARDS)
# ---------------------------------------------------------
MINT_CSS = """
<style>

    .eco-title {
        font-size: 40px;
        font-weight: 700;
        color: #2f5c3b;
        text-align: center;
        margin-top: 10px;
    }

    .eco-sub {
        font-size: 18px;
        text-align: center;
        color: #3c6b47;
        margin-bottom: 25px;
    }

    .eco-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,100,50,0.15);
        margin-bottom: 25px;
    }

    .stButton>button {
        background-color: #4caf75;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 16px;
        border: none;
        width: 100%;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
        transition: 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #5ed08c;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.25);
    }

    .result-box {
        background: #e8ffed;
        border-left: 8px solid #4caf75;
        padding: 18px;
        border-radius: 12px;
        color: #2f5c3b;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
        font-weight: 600;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
    }

</style>
"""
st.markdown(MINT_CSS, unsafe_allow_html=True)

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
# FORM
# ---------------------------------------------------------
st.markdown("<div class='eco-card'>", unsafe_allow_html=True)

with st.form("input_form"):
    st.subheader("🌱 Enter your lifestyle details")

    col1, col2 = st.columns(2)
    inputs = {}

    for i, c in enumerate(CAT_COLS):
        options = sorted(train_df[c].dropna().unique().tolist())
        if i % 2 == 0:
            inputs[c] = col1.selectbox(f"🍃 {c}", options)
        else:
            inputs[c] = col2.selectbox(f"🌿 {c}", options)

    for i, n in enumerate(NUM_COLS):
        default_val = float(train_df[n].median())
        if i % 2 == 0:
            inputs[n] = col1.number_input(f"🔢 {n}", value=default_val)
        else:
            inputs[n] = col2.number_input(f"🔢 {n}", value=default_val)

    submit = st.form_submit_button("Estimate Carbon Footprint")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICTION + SUGGESTIONS
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

        st.write("")

        # ---------------------- SUGGESTIONS BASED ON RANGE ----------------------
        if pred < 1000:
            st.markdown("""
            ### 🌿 Low Emissions — Great Job!
            - Maintain sustainable transport habits  
            - Keep electricity consumption low  
            - Continue recycling regularly  
            - Practice mindful consumption  
            - Inspire others to adopt eco-friendly habits  
            """)

        elif 1000 <= pred < 2000:
            st.markdown("""
            ### 🌱 Suggestions for Improvement
            - Use public transport or carpool more often  
            - Shorten shower duration  
            - Switch to LED bulbs  
            - Start recycling plastic and metal  
            - Add plant-based meals to your diet  
            """)

        elif 2000 <= pred < 3000:
            st.markdown("""
            ### 🌼 Moderate Emissions — Needs Attention
            - Reduce private vehicle usage  
            - Use energy-efficient appliances  
            - Reduce AC/refrigerator usage  
            - Avoid single-use plastics  
            - Reduce frequency of flights  
            - Buy fewer new clothes (reduce fast fashion impact)  
            """)

        elif 3000 <= pred < 4000:
            st.markdown("""
            ### 🌻 High Emissions — Consider These Steps
            - Shift to renewable energy (solar/green power)  
            - Reduce diesel/petrol usage  
            - Reduce packaged food & meat-heavy diet  
            - Conduct home energy audit  
            - Compost to reduce waste  
            - Reduce excessive screen time  
            """)

        else:
            st.markdown("""
            ### 🔥 Very High Emissions — Immediate Action Needed
            - Switch to electric/hybrid transport  
            - Install solar panels / switch from coal heating  
            - Avoid unnecessary flights  
            - Reduce fast fashion shopping  
            - Improve home insulation  
            - Cut down AC/heating use  
            """)

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)

