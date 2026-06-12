import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/sales.csv")

# Train model
X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

# App title
st.title("📈 Predictive Sales Forecast")

st.write("Predict future sales using Machine Learning.")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Input month
month = st.number_input("Enter Future Month", min_value=1, step=1)

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict([[month]])
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

# Sales trend chart
st.subheader("Sales Trend")

fig, ax = plt.subplots()
ax.plot(df["Month"], df["Sales"], marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Historical Sales")

st.pyplot(fig)