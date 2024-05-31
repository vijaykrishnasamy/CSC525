import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from a local drive
boston_data = pd.read_csv("BostonHousingData.csv")

# Display the first few rows of the dataset
print("Display the first few rows of the dataset: \n")
print(boston_data.head())
print("\n")

# Display basic information about the dataset
print("Display basic information about the dataset: \n")
print(boston_data.info())
print("\n")

# Check for missing values
print("Check for missing values: \n")
print(boston_data.isnull().sum())
print("\n")

# Display summary statistics
print("Display summary statistics: \n")
print(boston_data.describe())
print("\n")

# Separate the features and the target variable
X = boston_data.drop('MEDV', axis=1)
y = boston_data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predicting the prices of the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print("\n")

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Predict the housing price for a new data
new_data = np.array([[0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.90, 9.14]])
predicted_price = model.predict(new_data)

print(f'Predicted Housing Price: {predicted_price[0]}')
