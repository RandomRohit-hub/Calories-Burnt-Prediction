import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


st.title("🔥 Calorie Burn Prediction App")

st.markdown("""
<style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    .stForm label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load local CSVs
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    merged = pd.concat([exercise, calories['Calories']], axis=1)
    if 'User_ID' in merged.columns:
        merged.drop('User_ID', axis=1, inplace=True)
    return merged

# Load data
calories_data = load_data()

# Preprocessing
calories_data['Gender'] = calories_data['Gender'].map({'male': 0, 'female': 1})

X = calories_data.drop('Calories', axis=1)
Y = calories_data['Calories']

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train Model
model = XGBRegressor()
model.fit(X_train, Y_train)

# 🔮 Prediction Section
st.header("🎯 Try It Out: Predict Your Calorie Burn")

with st.form("user_input_form"):
    gender = st.selectbox("👤 Gender", ["male", "female"])
    age = st.slider("🎂 Age", 10, 80, 25)
    height = st.slider("📏 Height (cm)", 100, 220, 170)
    weight = st.slider("⚖️ Weight (kg)", 30, 150, 70)
    duration = st.slider("⏱️ Exercise Duration (minutes)", 5, 180, 30)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 200, 100)
    body_temp = st.slider("🌡️ Body Temperature (°C)", 35.0, 42.0, 37.0)
    submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_df = pd.DataFrame({
            'Gender': [0 if gender == 'male' else 1],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp]
        })

        input_df = input_df[X.columns]  # Ensure same order as training data
        pred = model.predict(input_df)
        st.success(f"🔥 Estimated Calories Burned: {pred[0]:.2f} kcal")

# Display basic info
st.subheader("🧾 Merged Dataset Preview")
st.dataframe(calories_data.head())

st.subheader("📊 Data Description")
st.write(calories_data.describe())

# Visualizations
st.subheader("📈 Visualizations")
sns.set()

fig1, ax1 = plt.subplots()
sns.countplot(data=calories_data, x='Gender', ax=ax1)
st.pyplot(fig1)

fig2 = sns.displot(calories_data['Age'])
st.pyplot(fig2.figure)

fig3, ax3 = plt.subplots()
sns.histplot(calories_data['Height'], kde=True, ax=ax3)
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
sns.histplot(calories_data['Weight'], kde=True, ax=ax4)
st.pyplot(fig4)

# Predict and Evaluate
prediction = model.predict(X_test)
r2 = metrics.r2_score(Y_test, prediction)
mae = metrics.mean_absolute_error(Y_test, prediction)

st.subheader("📈 Model Performance")
st.write(f"✅ R² Score: {r2:.2f}")
st.write(f"📉 Mean Absolute Error: {mae:.2f}")
