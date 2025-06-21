# electricity_predictor.py
# Predict monthly electricity bill based on units consumed using Linear Regression

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Step 1: Prepare sample data
data = {
    'Units': [100, 150, 200, 250, 300],
    'Bill': [500, 750, 1000, 1250, 1500]
}

# Step 2: Convert data to a DataFrame
df = pd.DataFrame(data)

# Step 3: Separate input and output
X = df[['Units']]  # Independent variable (features)
y = df['Bill']     # Dependent variable (target)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Take user input
try:
    units = float(input("Enter the number of electricity units consumed (kWh): "))
except ValueError:
    print("Invalid input! Please enter a numeric value.")
    exit()

# Step 6: Predict the electricity bill
predicted_bill = model.predict(np.array([[units]]))

# Step 7: Show the result
print(f"Estimated Electricity Bill: Tk {predicted_bill[0]:,.2f}")