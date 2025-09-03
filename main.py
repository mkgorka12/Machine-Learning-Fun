# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ml_components import LinearRegressionModel

data_file_path = 'Car_Price_Prediction.csv'

# Upload data into DataFrame
df = pd.read_csv(data_file_path)
print(df.head(int(len(df) * 0.2)), end='\n\n') # Print first 20% of data

# Show basic metadata
print(df.describe(), end='\n\n') # Print basic metadata
print(df.corr(numeric_only = True), end='\n\n') # Print correlation matrix

# Visualise data
pd.plotting.scatter_matrix(df, figsize=(8, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# Prepare data
x = df[['Year', 'Engine Size', 'Mileage']].to_numpy()
y = df[['Price']].to_numpy()

x_len = x.shape[0]
y_len = y.shape[0]

test_data_ratio = 0.2

x_train = x[:-int(x_len * test_data_ratio)]
x_test = x[int(x_len * test_data_ratio):]

y_train = y[:-int(y_len * test_data_ratio)]
y_test = y[int(y_len * test_data_ratio):]

# Train models
scikit_model = LinearRegression().fit(x_train, y_train)
own_model = LinearRegressionModel().fit(x_train, y_train, number_epochs=1000)

# Compare results
scikit_models_prediction = scikit_model.predict(x_test)
own_models_prediction = own_model.predict(x_test)

print(f"Scikit's prediction: {scikit_models_prediction}")
print(f"Author model's prediction: {own_models_prediction}")
print(f"Difference between models' prediction: {abs(scikit_models_prediction - own_models_prediction)}")
