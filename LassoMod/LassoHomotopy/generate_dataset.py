import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

np.random.seed(42)

# Generate synthetic dataset
X, y, true_coefficients = make_regression(
    n_samples=100,         
    n_features=20,
    n_informative=5,
    noise=0.1,
    coef=True, 
    random_state=42
)

# Introduce collinearity: make some features nearly identical
X[:, 10] = X[:, 0] + np.random.normal(0, 0.01, size=X[:, 0].shape) 
X[:, 11] = X[:, 1] + np.random.normal(0, 0.01, size=X[:, 1].shape)
X[:, 12] = X[:, 2] + np.random.normal(0, 0.01, size=X[:, 2].shape)

# Convert X to a DataFrame with column names
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# Convert y to a DataFrame with a single column
y_df = pd.DataFrame(y, columns=["target"])

# Convert true coefficients to a DataFrame
coef_df = pd.DataFrame(true_coefficients, index=[f"feature_{i}" for i in range(len(true_coefficients))], columns=["true_coefficient"])

# Save to CSV files
X_df.to_csv('data/test_data_X.csv', index=False)
y_df.to_csv('data/test_data_y.csv', index=False)
coef_df.to_csv('data/true_coefficients.csv', index=True)

print("Dataset generated and saved as CSV to 'data/' directory.")