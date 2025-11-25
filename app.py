import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os

# ================= 🌿 GEMINI CONFIGURATION =================
# ⚠️ REPLACE WITH YOUR KEY

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_suggestions(user_inputs, emission_value):
    prompt = f"""
    You are EcoScan AI — a friendly sustainability expert.

    The user has the following lifestyle choices:
    {user_inputs}

    Their estimated carbon footprint is: {emission_value:.2f} kg CO2 / month.

    ✨ Provide exactly 5 short, practical tips to reduce footprint.
    📌 Rules:
    - Max 10–12 words per point.
    - Relevant to given lifestyle only.
    - Simple language, no technical words.
    - Encouraging, not blaming.
    - Format with bullet points like this:
      • suggestion 1
      • suggestion 2
      • suggestion 3
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


# ================= 🌿 STREAMLIT UI DESIGN =================
st.set_page_config(page_title="EcoScan", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
        :root { color-scheme: light !important; }
        .stApp { background-color: #0B3D2E !important; }
        .eco-title { font-size: 40px; font-weight: 700; color: #FFFFFF; text-align: center; margin-top: 10px; }
        .eco-sub { font-size: 18px; text-align: center; color: #D9FFEE; margin-bottom: 25px; }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
        label, .stMarkdown p { color: #E8FFF4 !important; font-weight: 600 !important; }
        .eco-card { background: #ffffff; border-radius: 16px; padding: 22px 26px; box-shadow: 0px 6px 18px rgba(0,0,0,0.3); margin-bottom: 25px; }
        .stButton>button {
            background-color: #157F56; color: white; border-radius: 10px;
            padding: 10px 18px; font-size: 16px; border: none; width: 100%;
            box-shadow: 0px 3px 12px rgba(0,0,0,0.3); transition: 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1CA56C; box-shadow: 0px 5px 18px rgba(0,0,0,0.4);
        }
        .result-box {
            background: #E8FFF0; border-left: 8px solid #1CA56C; padding: 18px;
            border-radius: 12px; color: #0B3D2E; font-size: 20px; text-align: center;
            margin-top: 20px; font-weight: 600; box-shadow: 0px 4px 14px rgba(0,0,0,0.25);
        }
        /* Force ALL markdown text to white, including Gemini suggestions */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }

        /* Light color bullet dot */
        .stMarkdown li::marker {
            color: #D9FFEE !important;
        }

    </style>
""", unsafe_allow_html=True)


# ================= 🌿 LOAD MODEL + DATA =================
MODEL_PATH = "artifacts/model.pkl"
TRAINING_CSV_PATH = "data/Cleaned_Carbon_Emission.csv"

@st.cache_resource
def load_pipeline_model():
    return joblib.load(MODEL_PATH)   # contains preprocessor + regressor

def load_training_data():
    return pd.read_csv(TRAINING_CSV_PATH)

model = load_pipeline_model()
train_df = load_training_data()

# Extract columns for UI only (NOT preprocessing)
NUM_COLS = train_df.select_dtypes(exclude="object").columns.drop("CarbonEmission", errors="ignore").tolist()
CAT_COLS = train_df.select_dtypes(include="object").columns.tolist()


# ================= 🌿 APP TITLE =================
st.markdown("<div class='eco-title'>🌿 EcoScan — Carbon Footprint Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='eco-sub'>Make sustainable decisions with data-driven insights</div>", unsafe_allow_html=True)


# ================= 🌿 FORM UI =================
st.markdown("<div class='eco-card'>", unsafe_allow_html=True)

with st.form("input_form"):
    st.subheader("🌱 Enter your lifestyle details")

    col1, col2 = st.columns(2)
    inputs = {}

    # Categorical inputs
    for i, c in enumerate(CAT_COLS):
        options = sorted(train_df[c].dropna().unique().tolist())
        if i % 2 == 0:
            inputs[c] = col1.selectbox(f"🍃 {c}", options)
        else:
            inputs[c] = col2.selectbox(f"🌿 {c}", options)

    # Numeric inputs
    for i, n in enumerate(NUM_COLS):
        default_val = float(train_df[n].median())
        if i % 2 == 0:
            inputs[n] = col1.number_input(f"🔢 {n}", value=default_val)
        else:
            inputs[n] = col2.number_input(f"🔢 {n}", value=default_val)

    submit = st.form_submit_button("Estimate Carbon Footprint")

st.markdown("</div>", unsafe_allow_html=True)


# ================= 🌿 PREDICT + GEMINI =================
if submit:
    try:
        input_df = pd.DataFrame([inputs])

        # Convert list inputs if any to strings
        for col in input_df.columns:
            if isinstance(input_df[col][0], list):
                input_df[col] = input_df[col].apply(lambda x: ", ".join(x))

        # Direct prediction (pipeline handles preprocessing)
        pred = model.predict(input_df)[0]

        st.markdown(
            f"<div class='result-box'>🌎 Your estimated carbon footprint:<br><br><b>{pred:.2f} kg CO₂ / month</b></div>",
            unsafe_allow_html=True
        )

        suggestions = generate_suggestions(inputs, pred)
        st.markdown("### 🌿 Personalized Eco Suggestions")
        st.markdown(suggestions)

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)
