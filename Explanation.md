# SA House Price Predictor - Project Explanation

This document provides a detailed overview of the South African House Price Predictor project, including its structure, functionality, usage, and the development process.

## 1. File Structure

```
SAHousePricePredictor/
├── house_prices_train.csv      # Training dataset (Johannesburg housing data)
├── house_prices_test.csv       # Testing dataset
├── requirements.txt            # Python dependencies
├── model.joblib                # Trained Linear Regression model (serialized)
├── Explanation.md              # This documentation file
└── src/
    ├── data_loader.py          # Utility to load CSV data
    ├── eda.py                  # Script for Exploratory Data Analysis
    ├── check_columns.py        # Utility to inspect dataset columns
    ├── pipeline.py             # Main processing and training pipeline
    ├── api.py                  # FastAPI backend for predictions
    ├── app.py                  # Streamlit frontend for user interaction
    └── test_api.py             # Script to test the API endpoint
```

## 2. File Descriptions

- **`src/data_loader.py`**: Contains the `load_data` function which safely reads the train and test CSV files, handling file path validation.
- **`src/pipeline.py`**: The core machine learning script. It:
    - Cleans the data (drops rows with missing prices, parses 'Erf Size').
    - Defines a Scikit-Learn `Pipeline` with `ColumnTransformer` for preprocessing (StandardScaler for numerics, OneHotEncoder for categoricals).
    - Trains a `LinearRegression` model.
    - Evaluates the model on a validation set.
    - Saves the trained model to `model.joblib`.
- **`src/api.py`**: A REST API built with FastAPI. It loads the saved model and exposes a `/predict` endpoint that accepts property details and returns a price estimate.
- **`src/app.py`**: A web-based user interface built with Streamlit. It provides a form for users to input property attributes, sends them to the API, and displays the result.
- **`src/eda.py`**: A script used during development to inspect the data distribution, check for missing values, and understand feature types.

## 3. How to Run

### Prerequisites
Ensure you have Python installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model
To retrain the model (e.g., if data changes or code is modified):
```bash
python src/pipeline.py
```
This will output performance metrics and overwrite `model.joblib`.

### Running the API
- **Tech Stack**: Python, Pandas (data), Scikit-Learn (ML), FastAPI (Backend), Streamlit (Frontend).

### Phase 2: Data Exploration (EDA)
- We inspected the data using `src/eda.py`.
- **Findings**:
    - `Location` column contained a single repeated value, so it was dropped as it provided no information.
    - `Erf Size` contained units (m²) and needed parsing to be numeric.
    - `Floor Size` was missing from the dataset despite being a common feature, so it was excluded.
    - Target variable `Price` had some missing values (NaN), which were dropped.

### Phase 3: Model Building
- **Preprocessing**:
    - Numeric features (`Bedrooms`, `Bathrooms`, `Erf Size`) are imputed (median) and scaled.
    - Categorical features (`Type of Property`) are imputed (mode) and one-hot encoded.
- **Algorithm**: We used Linear Regression as a baseline. It provides good interpretability and a solid starting point.
- **Validation**: The model was evaluated on a hold-out test set (or a split of the training set) to ensure it generalizes well.

### Phase 4: Deployment (API & UI)
- **API**: Wrapped the model in a FastAPI application to allow external systems to request predictions.
- **UI**: Built a simple, interactive Streamlit dashboard to make the tool accessible to end-users without coding knowledge.
