# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from ml_components import Model, Hyperparameters

# Upload data into DataFrame
car_pricing_df = pd.read_csv('Car_Price_Prediction.csv')
print(car_pricing_df.head(len(car_pricing_df) // 5), end='\n\n')

# Show basic metadata
print(car_pricing_df.describe(), end='\n\n')
print(car_pricing_df.corr(numeric_only = True), end='\n\n') # Correlation matrix

# Visualise our data
pd.plotting.scatter_matrix(car_pricing_df, figsize=(8, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# Init model
hyperparameters = Hyperparameters(batch_size=100, number_epochs=200)
model = Model(car_pricing_df['Price'].to_list(), car_pricing_df['Mileage'].to_list(), hyperparameters)

# Train model
model.train(plot_losses=True)

# Test prediction
test_feature = 97000
print(f"Feature: {test_feature}, Model's prediction: {model.predict(test_feature)}")
