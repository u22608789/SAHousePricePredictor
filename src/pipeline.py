import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from data_loader import load_data

def parse_erf_size(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace(' m²', '').replace(' ', '')
        # Handle hectares if present (e.g., "1.5 ha") - though EDA showed m²
        if 'ha' in value:
            return float(value.replace('ha', '')) * 10000
        try:
            return float(value)
        except ValueError:
            return np.nan
    return float(value)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame.
    """
    df = df.copy()
    
    # Drop Location as it appears constant/useless based on EDA
    if 'Location' in df.columns:
        df = df.drop(columns=['Location'])
        
    # Drop ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Parse Erf Size
    if 'Erf Size' in df.columns:
        df['Erf Size'] = df['Erf Size'].apply(parse_erf_size)

    # Convert Listing Date to datetime and extract features (e.g., Year, Month)
    # For now, we might just drop it or use it to calculate age if we had built year.
    # Let's just drop it for the baseline model as it might not be very predictive without more context.
    if 'Listing Date' in df.columns:
        df = df.drop(columns=['Listing Date'])

    # Drop rows where Price is NaN
    if 'Price' in df.columns:
        df = df.dropna(subset=['Price'])

    return df

def build_pipeline():
    """
    Builds the Scikit-Learn pipeline.
    """
    numeric_features = ['Bedrooms', 'Bathrooms', 'Erf Size']
    categorical_features = ['Type of Property']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    return model

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "house_prices_train.csv")
    test_file = os.path.join(base_dir, "house_prices_test.csv")

    print("Loading data...")
    train_df, test_df = load_data(train_file, test_file)

    print("Cleaning data...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # Separate target
    X_train = train_df.drop(columns=['Price'])
    y_train = train_df['Price']
    
    # For test set, we also have Price? The CSV name implies it might be a holdout without labels, 
    # but usually 'test' in kaggle has no labels. Let's check if Price is in test_df.
    if 'Price' in test_df.columns:
        X_test = test_df.drop(columns=['Price'])
        y_test = test_df['Price']
    else:
        # If no price in test, we can't evaluate on it. We'll split train_df for validation.
        print("No Price column in test set. Splitting train set for validation.")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("Training model...")
    print(f"X_train columns: {X_train.columns.tolist()}")
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Metrics ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    model_path = os.path.join(base_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
