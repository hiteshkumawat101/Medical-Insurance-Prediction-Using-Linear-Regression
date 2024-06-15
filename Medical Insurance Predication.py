# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing necessary modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load the dataset
insurance_data = pd.read_csv('C:/Users/Hitesh/Downloads/insurance.csv')

# Display the first few rows of the dataset
insurance_data.head()

# Display the shape of the dataset (number of rows and columns)
insurance_data.shape

# Display basic statistical details of the dataset
insurance_data.describe()

# Check for missing values in the dataset
insurance_data.isna().sum()

# Visualize average charges by sex using a bar plot
sns.barplot(x='sex', data=insurance_data, y='charges')

# Set the aesthetic style of the plots and plot the distribution of ages
sns.set()
sns.distplot(insurance_data['age'])

# Visualize average charges by the number of children using a bar plot
sns.barplot(x='children', data=insurance_data, y='charges')

# Display the count of each number of children
insurance_data['children'].value_counts()

# Visualize average charges by smoking status using a bar plot
sns.barplot(x='smoker', data=insurance_data, y='charges')

# Visualize average charges by region using a bar plot
sns.barplot(x='region', data=insurance_data, y='charges')

# Display the count of each region
insurance_data['region'].value_counts()

# Encode categorical variables into numerical values
insurance_data.replace({
    'sex': {'male': 0, 'female': 1},
    'smoker': {'yes': 1, 'no': 0},
    'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}
}, inplace=True)

# Display the first few rows of the modified dataset
insurance_data.head()

# Separate features (X) and target variable (Y)
X = insurance_data.drop('charges', axis=1)
print(X.head())
Y = insurance_data['charges']
Y.head()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=5, test_size=0.3)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict charges using the training data and evaluate the model
p1 = model.predict(x_train)
print(f'R^2 score for training data: {metrics.r2_score(y_train, p1)}')

# Predict charges using the testing data and evaluate the model
p2 = model.predict(x_test)
print(f'R^2 score for testing data: {metrics.r2_score(y_test, p2)}')

# Example: Predicting charges for a new data point
inpu = (31, 1, 25.74, 0, 1, 0)
inpu = np.asarray(inpu)
inpu = inpu.reshape(1, -1)
print(f'Input array: {inpu}')

# Predict charges for the given input
predicted_charges = model.predict(inpu)
print(f'Predicted charges: {predicted_charges[0]}')


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the scaled training data
model.fit(x_train_scaled, y_train)

# Predict charges using the scaled training data and evaluate the model
p3 = model.predict(x_train_scaled)
print(f'R^2 score for training data: {metrics.r2_score(y_train, p3)}')

# Predict charges using the scaled testing data and evaluate the model
p4 = model.predict(x_test_scaled)
print(f'R^2 score for testing data: {metrics.r2_score(y_test, p4)}')

# Example: Predicting charges for a new data point
inp = (31, 1, 25.74, 0, 1, 0)
inp = np.asarray(inp)
inp = inp.reshape(1, -1)

# Scale the new input data using the same scaler
inpu_scaled = scaler.transform(inpu)
print(f'Input array: {inpu_scaled}')

# Predict charges for the given input
predicted_charges = model.predict(inpu_scaled)
print(f'Predicted charges: {predicted_charges[0]}')