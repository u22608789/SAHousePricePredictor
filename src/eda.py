import pandas as pd
from data_loader import load_data
import os

def run_eda():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "house_prices_train.csv")
    test_file = os.path.join(base_dir, "house_prices_test.csv")
    
    try:
        train, test = load_data(train_file, test_file)
        
        print("--- Train Info ---")
        print(train.info())
        print("\n--- Train Describe ---")
        print(train.describe())
        print("\n--- Train Head ---")
        print(train.head())
        
        print("\n--- Missing Values ---")
        print(train.isnull().sum())
        
        print("\n--- Columns ---")
        print([repr(c) for c in train.columns])

        # Check for non-numeric columns
        print("\n--- Object Columns ---")
        for col in train.select_dtypes(include=['object']).columns:
            print(f"\nValue Counts for {col}:")
            print(train[col].value_counts().head(10))

    except Exception as e:
        print(f"Error in EDA: {e}")

if __name__ == "__main__":
    run_eda()
