import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

dataset = {
    'Matches': [13,16,16,16,16,16,14,16,16,10,14,14,15,15,16,14],
    'Strike Rate': [105.09,112.32,144.81,121.08,111.65,138.73,122.10,130.82,152.03,122.22,139.10,141.46,121.35,119.46,115.99,139.82],
    'Runs': [165,246,307,557,364,634,359,505,973,308,530,464,466,405,341,639],
    'Average': [15.00,22.36,27.90,46.41,28.00,45.28,27.61,45.90,81.08,30.80,48.18,33.14,42.36,28.29,22.73,53.25],
    'Balls Faced': [157,219,212,460,326,457,294,386,640,252,381,328,384,339,294,457],
    'Year': [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'Age': [19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
}
df = pd.DataFrame(dataset)
# print(df)
# Features for prediction
X = df[['Year']]
# Target feature
y = df[['Runs', 'Strike Rate']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
# model = LinearRegression()
rf_model = RandomForestRegressor()
model = MultiOutputRegressor(rf_model)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Example: Predict 'Runs' for a specific year
input_year = int(input("Enter the year to predict runs: "))  # Replace with the desired year
prediction = model.predict([[input_year]])
print("Prediction for Runs in {}: {}".format(input_year, int(prediction[0,0])))
print("Prediction for Strike Rate {}: {}".format(input_year,float(prediction[0,1])))