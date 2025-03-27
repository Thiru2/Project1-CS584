#              Team Members
#    THIRUMALESH KURUKUNDHA  -- A20561775
#    MAHENDRA REDDY KADIRE   -- A20549975
#    KARTHIK KALLURI         -- A20552245



# Lasso Homotopy Project

This project implements a Lasso regression model using the Homotopy method, with tests to verify sparsity, collinearity handling, prediction accuracy, and convergence. Below are instructions to set up and run the project on Windows, macOS, and Linux.

## Prerequisites

- **Python 3**: Ensure Python 3 is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
- **pip3**: Comes with Python 3; used to install dependencies.
- **Terminal/Command Prompt**: Access to a command-line interface.

## Project Structure

- `models/lasso.py`: Contains the `LassoHomotopy` class implementation.
- `tests/test_lasso.py`: Test suite for the Lasso model.
- `data/`: Directory with `test_data_X.csv`, `test_data_y.csv`, and `true_coefficients.csv`.
- `requirements.txt`: Lists project dependencies.

## Setup Instructions

Follow these steps to set up and run the project. Instructions are provided for Windows, macOS, and Linux.

### 1. Clone or Download the Project
- If using Git: `git clone <repository-url>`
- Otherwise, download and extract the project folder to a location of your choice (e.g., `C:\Projects\LassoHomotopy` on Windows or `~/Projects/LassoHomotopy` on macOS/Linux).

### 2. Create a Virtual Environment

A virtual environment isolates project dependencies. Use Python 3’s `venv` module.

#### Windows
1. Open Command Prompt (cmd) or PowerShell.
2. Navigate to the project directory:
   ```bash
   cd C:\Projects\LassoHomotopy
   ```
3. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Command Prompt:
     ```bash
     venv\Scripts\activate
     ```
   - PowerShell:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   You’ll see `(venv)` in your prompt.

#### macOS/Linux
1. Open Terminal.
2. Navigate to the project directory:
   ```bash
   cd ~/Projects/LassoHomotopy
   ```
3. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   You’ll see `(venv)` in your prompt.

### 3. Install Dependencies

With the virtual environment activated, install the required packages from `requirements.txt`.

#### All Operating Systems
Ensure you’re in the project root directory (where `requirements.txt` is located) and run:
```bash
pip3 install -r requirements.txt
```

This installs:
- numpy
- pandas
- pytest
- ipython
- scikit-learn

### 4. Verify Installation

Check installed packages:
```bash
pip3 list
```
You should see numpy, pandas, pytest, ipython, and scikit-learn listed.

### 5. Navigate to the Correct Folder

The tests assume the working directory is the project root (where `tests/` and `models/` are subdirectories). If you’re not already there:
```bash
cd <path-to-LassoHomotopy>
```
Examples:
- Windows: `cd C:\Projects\LassoHomotopy`
- macOS/Linux: `cd ~/Projects/LassoHomotopy`

### 6. Run the Tests

Run the test suite using `pytest`.

#### MacOS/Windows Operating Systems
From the project root, execute the following command for mac:
```bash
PYTHONPATH=. pytest tests/
```

execute the following command for windows:
```bash
$env:PYTHONPATH="." ; pytest tests/
```
`PYTHONPATH=.` ensures Python finds the `models` module.
`pytest tests/` runs all tests in the `tests/` directory.

#### Expected Output
You’ll see a summary like:
```bash
collected 4 items
tests/test_lasso.py .... [100%]
4 passed in X.XXs
```
If any test fails, the output will detail the failure (e.g., assertion errors with values).

## Model and Tests

### What Does the Model Do and When Should It Be Used?
The `LassoHomotopy` class implements Lasso regression using the Homotopy method. Lasso is a linear regression technique that applies an L1 penalty to promote sparsity, setting some coefficients to zero for feature selection while fitting the data. The Homotopy method computes the solution path by decreasing the regularization parameter (`lambda_`) from a maximum to a minimum, efficiently tracking coefficient changes.

- **Purpose**: It performs simultaneous feature selection and regression, ideal for identifying key predictors in high-dimensional data.
- **When to Use**: Use it when you have many features, suspect some are irrelevant or redundant, or need a sparse, interpretable model. It’s particularly useful with collinear features, as it selects one from correlated groups. Example: Predicting outcomes with numerous potential predictors (e.g., gene expression analysis).
- **Expected Outcome**: A sparse coefficient vector (many zeros) that minimizes prediction error, highlighting the most influential features.

### How Did You Test the Model?
The model’s correctness is validated through four tests in `tests/test_lasso.py` using `pytest`, run on synthetic data (`data/test_data_X.csv`, `test_data_y.csv`, `true_coefficients.csv`):

1. **`test_lasso_sparsity`**:
   - **Check**: Ensures at least 8 of 20 coefficients are zero.
   - **Outcome**: Passes if `np.sum(model.coef_ == 0) >= 8`, confirming sparsity.

2. **`test_lasso_collinearity`**:
   - **Check**: For collinear pairs (0-10, 1-11), coefficient differences are < 0.01 or higher-index coefficients (10, 11) are 0.
   - **Outcome**: Passes if collinearity is handled, mimicking Lasso’s feature selection behavior.

3. **`test_lasso_prediction`**:
   - **Check**: Mean Squared Error (MSE) between predictions and targets is < 1.0.
   - **Outcome**: Passes if MSE is less than 1.0, verifying predictive accuracy.

4. **`test_lasso_convergence`**:
   - **Check**: Coefficients are not `None` after fitting.
   - **Outcome**: Passes if `model.coef_ is not None`, ensuring convergence.

### Tunable Parameters
Users can tune these `LassoHomotopy` parameters:
- **`lambda_max`**: Initial regularization strength (default: `None`, auto-set). Higher values increase sparsity; lower values retain more features.
- **`lambda_min`**: Minimum regularization (default: 0.0001). Lower values allow finer solutions; higher values enforce sparsity.
- **`max_iter`**: Max iterations (default: 1000). Increase for complex data; decrease for speed.
- **`tol`**: Convergence tolerance (default: 1e-4). Smaller values enhance precision; larger values hasten convergence.

### Specific Inputs and Limitations
- **Troublesome Inputs**:
  - **High Noise**: Low signal-to-noise ratios may obscure true predictors, raising MSE.
  - **Extreme Collinearity**: Beyond predefined pairs, unhandled correlations can skew selection.
  - **Small Datasets**: Limited samples may lead to overfitting or unstable convergence.

- **Workarounds with Time**: 
  - Noise: Add cross-validation for `lambda_` tuning.
  - Collinearity: Dynamically detect correlations (e.g., via matrix analysis).
  - Small Data: Adjust regularization or use simpler models.

- **Fundamental Limits**: Lasso’s L1 penalty arbitrarily picks one feature from perfectly collinear sets, a core limitation. Elastic Net could mitigate this but requires reimplementation.

## Troubleshooting

- **Python3 not found**: Ensure Python 3 is installed and accessible. Try `python --version` or `python3 --version`. On some systems, `python` might be Python 2; use `python3` explicitly.
- **pip3 not found**: Install it with `python3 -m ensurepip --upgrade` then `python3 -m pip install --upgrade pip`.
- **ModuleNotFoundError**: Confirm `PYTHONPATH=.` is set and you’re in the project root.
- **Permission denied (macOS/Linux)**: Add `sudo` if needed (e.g., `sudo pip3 install -r requirements.txt`), but prefer user-level installs with `--user` if outside a venv.
- **Test data missing**: Ensure `data/test_data_X.csv`, `data/test_data_y.csv`, and `data/true_coefficients.csv` exist in the `data/` directory.

### Deactivating the Virtual Environment
When done, deactivate the virtual environment:
```bash
deactivate
```

## Additional Notes
- **Python Version**: The project uses Python 3.10.0rc2 (from your output), but any Python 3.x should work.
- **Dependencies**: `scikit-learn` isn’t directly used in the provided code but is included for potential future extensions.
- **Running IPython**: For interactive exploration, run `ipython` in the activated venv and import `LassoHomotopy` from `models.lasso`.