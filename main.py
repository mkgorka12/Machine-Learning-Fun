import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from linear_model import LinearRegression as MlFunLinearRegression

from time import time_ns

welcome_msg = 'COMPARE ML FUN LINEAR REGRESSION MODEL TO SCIKIT'
print(f"{ welcome_msg }\n{ len(welcome_msg)*'#' }\n")

filepath = input('Enter the filepath to the .csv file: ')

# Upload data into DataFrame
try:
    df = pd.read_csv(filepath).select_dtypes(include='number') # Numeric values only
except:
    print(f'No such file or directory: "{ filepath }"')
    exit(1)

print(f'\n{df.head()}\n')

# Show basic metadata
print(df.describe(), end='\n\n') # Print basic metadata
df_corr = df.corr(numeric_only=True)
print(df_corr, end='\n\n') # Print correlation matrix

label = input('What label do you want to train?: ')
if label not in df.columns:
    print(f'No label named "{label}" in the DataFrame')
    exit(2)

n = int(input('How many most correlated features do you want to use?: '))
if n < 0 or n >= df.shape[1]:
    print(f'Invalid number of features')
    exit(3)

print('')

# Find top n most correlated features to the label
corr_with_label = df_corr[label].drop(label)
top_n_corr = corr_with_label.abs().sort_values(ascending=False).head(n).to_dict()
top_n_corr_names = top_n_corr.keys()

# Visualise data
pd.plotting.scatter_matrix(df[[*top_n_corr_names, label]], figsize=(8, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# Prepare data 
x = df[top_n_corr_names].to_numpy()
y = df[[label]].to_numpy()

x_len = x.shape[0]
y_len = y.shape[0]

test_to_train_ratio = 0.2 # 80% of the data is a training set, 20% left is a test set

x_train = x[:-int(x_len * test_to_train_ratio)]
x_test = x[int(x_len * test_to_train_ratio):]

y_train = y[:-int(y_len * test_to_train_ratio)]
y_test = y[int(y_len * test_to_train_ratio):]

# Train models
start = time_ns()
scikit_model = SklearnLinearRegression().fit(x_train, y_train)
training_time = (time_ns() - start) / 1000000

print(f"Scikit model trained in {training_time} ms")

start = time_ns()
author_model = MlFunLinearRegression().fit(x_train, y_train)
training_time = (time_ns() - start) / 1000000

print(f"ML-Fun model trained in {training_time} ms", end='\n\n')

# Compare results
scikit_model_prediction = scikit_model.predict(x_test)
author_model_prediction = author_model.predict(x_test) 

print(f"Scikit model's RMSE: {root_mean_squared_error(y_test, scikit_model_prediction)}")
print(f"ML-Fun model's RMSE {root_mean_squared_error(y_test, author_model_prediction)}")
