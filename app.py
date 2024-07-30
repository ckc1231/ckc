import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Streamlit sidebar inputs
st.sidebar.title("Linear Regression Parameters")
a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
c = st.sidebar.slider("Noise Level (c)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
n = st.sidebar.slider("Number of Sample Points (n)", min_value=100, max_value=1000, value=200, step=10)

# Generate sample points
x_values = np.random.uniform(-100, 100, n)
noise = np.random.normal(0, 1, n)
y_values = a * x_values + 50 + c * noise

# Perform linear regression
x_values_reshaped = x_values.reshape(-1, 1)
model = LinearRegression()
model.fit(x_values_reshaped, y_values)

# Predict y values
y_pred = model.predict(x_values_reshaped)

# Plot the points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', s=10, label='Data Points')
plt.plot(x_values, y_pred, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with User Inputs')
plt.legend()

# Display the plot
st.pyplot(plt)

# Display regression coefficients
st.write(f"Intercept: {model.intercept_}")
st.write(f"Coefficient (Slope): {model.coef_[0]}")
