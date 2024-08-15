import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Step 1: Load the dataset
file_path = 'sale.csv'  # Path to the CSV file
data = pd.read_csv(file_path)


# Display the first few rows of the dataset and the column names
print("Dataset preview:")
print(data.head())
print("\nColumn names:")
print(data.columns)


# Assuming 'Jan' is the column we want to predict
target_variable = 'Jan'


# Check if the target column exists
if target_variable not in data.columns:
    raise KeyError(f"Target column '{target_variable}' not found in dataset columns: {data.columns}")


# Step 2: Preprocess the data
# Handle missing values
data.filling(method='fill', inplace=True)


# Encode categorical variables if any
data = pd.get_dummies(data)


# Separate features and target variable
X = data.drop(target_variable, axis=1)
y = data[target_variable]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 3: Define and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 4: Evaluate the model
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# Step 5: Fine-tune the model (basic example)
# For more advanced models, hyperparameter tuning can be done using GridSearchCV or RandomizedSearchCV


# Step 6: Save the trained model
joblib.dump(model, 'linear_regression_model.joblib')
joblib.dump(scaler, 'scaler.joblib')


# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.grid(True)
plt.show()


# Save the results to a new CSV file
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('predictions.csv', index=False)
print("Results saved to predictions.csv")
