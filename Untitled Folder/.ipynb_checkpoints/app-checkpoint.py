
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load dataset
df = pd.read_csv("titanic_cleaned.csv")

# Load model (if available)
try:
    model = joblib.load("model.pkl")
except:
    model = None

# Apply Bootstrap
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Title and intro
st.markdown("<h1 class='text-center text-primary'>🛳 Titanic Survival Dashboard</h1>", unsafe_allow_html=True)
st.write("This interactive dashboard lets you explore the Titanic dataset and make survival predictions.")

# Sidebar filters
st.sidebar.header("Filter Data")
pclass = st.sidebar.selectbox("Passenger Class", options=[1, 2, 3])
sex = st.sidebar.radio("Sex", options=["male", "female"])
age = st.sidebar.slider("Age", min_value=0, max_value=80, value=30)
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)

# Filtered dataset display
st.subheader("Filtered Dataset")
filtered_df = df[(df['Pclass'] == pclass) & (df['Sex'] == sex)]
st.dataframe(filtered_df.head())

# Plot survival by sex and class
st.subheader("Survival Rate by Sex and Class")
fig = px.histogram(df, x="Sex", color="Survived", barmode="group", facet_col="Pclass", category_orders={"Pclass": [1,2,3]})
st.plotly_chart(fig)

# Prediction section
st.subheader("🎯 Predict Survival")
if model:
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [1 if sex == "female" else 0],
        "Age": [age],
        "Fare": [fare],
        "SibSp": [sibsp],
        "Parch": [parch]
    })
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    outcome = "✅ Survived" if prediction == 1 else "❌ Did not survive"
    st.markdown(f"<h3 class='text-success'>{outcome}</h3>", unsafe_allow_html=True)
    st.write(f"Survival Probability: {prob:.2%}")
else:
    st.warning("Model file not found. Please train and save your model as 'model.pkl'.")

st.markdown("---")
st.caption("Built with Streamlit + Bootstrap 💙")
