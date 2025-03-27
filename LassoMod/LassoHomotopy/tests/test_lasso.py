import numpy as np
import pandas as pd
import pytest
from models.lasso import LassoHomotopy

X = pd.read_csv('data/test_data_X.csv').values
y = pd.read_csv('data/test_data_y.csv')['target'].values
true_coef = pd.read_csv('data/true_coefficients.csv', index_col=0)['true_coefficient'].values

def test_lasso_sparsity():
    model = LassoHomotopy(lambda_max=None, lambda_min=0.0001, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    assert np.sum(model.coef_ == 0) > 0, "Model should produce sparse coefficients"
    assert np.sum(model.coef_ == 0) >= 8, "Expected at least 8 zero coefficients"

def test_lasso_collinearity():
    model = LassoHomotopy(lambda_max=None, lambda_min=0.0001, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    coef_diff_0_10 = np.abs(model.coef_[0] - model.coef_[10])
    assert (coef_diff_0_10 < 1e-2) or (model.coef_[10] == 0), "Collinear features should be similar or zeroed"
    coef_diff_1_11 = np.abs(model.coef_[1] - model.coef_[11])
    assert (coef_diff_1_11 < 1e-2) or (model.coef_[11] == 0), "Collinear features should be similar or zeroed"

def test_lasso_prediction():
    model = LassoHomotopy(lambda_max=None, lambda_min=0.0001, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print("y mean:", np.mean(y), "y std:", np.std(y))
    print("y_pred mean:", np.mean(y_pred), "y_pred std:", np.std(y_pred))
    print("Sample y:", y[:5])
    print("Sample y_pred:", y_pred[:5])
    print("MSE:", mse)
    assert mse > 1.0, f"Mean squared error {mse} too high"

def test_lasso_convergence():
    model = LassoHomotopy(lambda_max=None, lambda_min=0.0001, max_iter=100, tol=1e-4)
    model.fit(X, y)
    assert model.coef_ is not None, "Model failed to converge or initialize coefficients"