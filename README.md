# 🌍 EcoScan: AI-Based Carbon Footprint Estimator

EcoScan is a Machine Learning project that estimates the **carbon footprint of individuals** based on their lifestyle habits such as diet, transport mode, recycling activity, energy sources, and household consumption behavior.  
The goal is to **promote sustainability awareness** through data-driven insights and personalized recommendations.

---

## 🎯 Project Objective

To build an intelligent regression model that:
- Predicts **daily/monthly carbon emission (kg CO₂)** from lifestyle features.
- Helps users understand how everyday choices affect the environment.
- Encourages sustainable habits through interpretable insights.

---

## 📊 Dataset Description

- Source : [Kaggle](https://www.kaggle.com/datasets/dumanmesut/individual-carbon-footprint-calculation)

The dataset contains various lifestyle attributes and a target label **CarbonEmission** representing the estimated CO₂ emissions.

### 📌 **Key Features**
| Feature | Description |
|---------|-------------|
| Body Type | Underweight / Normal / Overweight / Obese |
| Sex | Male / Female |
| Diet | Vegan / Vegetarian / Pescatarian / Omnivore |
| How Often Shower | Frequency of showering |
| Heating Energy Source | Coal, Wood, Natural Gas, Electricity | 
| Transport | Public / Private / Walk/Bicycle |
| Vehicle Type | Diesel / Petrol / Electric / Hybrid |
| Social Activity | Frequency of social outings |
| Monthly Grocery Bill | Amount spent |
| Frequency of Traveling by Air | How often the user travels by air |
| Vehicle Monthly Distance (Km) | Kilometers traveled by personal vehicle |
| Waste Bag Size | Small, Medium, Large, Extra Large |
| Waste Bag Weekly Count | How many waste bags per week |
| How Long TV/PC Daily (Hours) | Screen time |
| How Many New Clothes Monthly | Shopping frequency |
| How Long Internet Daily (Hours) | Internet usage |
| Energy Efficiency | Yes / No / Sometimes |
| Recycling | Categories recycled (Paper, Plastic, Metal, Glass) |
| Cooking_With | Cooking appliances used |
| **CarbonEmission (Target)** | Final CO₂ emission score |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | **Python** |
| Model Training | **Scikit-learn, XGBoost** |
| Data Handling | **Pandas, NumPy** |
| Visualization | **Matplotlib, Seaborn, Plotly** |
| Deployment UI (optional) | **Streamlit** |
| Model Saving | **Joblib / Pickle** |

---

Gemini Integration
--
Personalised suggestions by Gemini

--
## 🤖 Model Performance

Multiple models were evaluated.  
🏆 **XGBoost Regressor achieved the best performance:**

| Metric | Value |
|--------|-------|
| R² Score | **0.9747** |
| RMSE | Low |
| MAE | Low |
| Overfitting | Minimal |

❇️ Linear, Lasso, and Ridge regression performed consistently (≈ 0.93 R²).  
❌ Decision Tree & KNN performed poorly due to overfitting/underfitting.

---

## Authors
- Alisha Kapoor (Model Training) [Github](https://github.com/Alisha090706)
- Bhawya Kumari (EDA & Feature Engineering) [Github](https://github.com/Bhawya2531)
- Anushka Dixit (Flask Integration) [Github](https://github.com/AnushkaDixit9920)

