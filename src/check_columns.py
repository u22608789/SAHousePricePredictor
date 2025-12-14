import pandas as pd
from data_loader import load_data
import os

def check_columns():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "house_prices_train.csv")
    test_file = os.path.join(base_dir, "house_prices_test.csv")
    
    train, _ = load_data(train_file, test_file)
    
    required = ['Bedrooms', 'Bathrooms', 'Floor Size', 'Erf Size', 'Type of Property']
    
    print("--- Column Check ---")
    for col in required:
        if col in train.columns:
            print(f"'{col}': FOUND")
        else:
            print(f"'{col}': MISSING")
            
    print("\nAll columns:", train.columns.tolist())

if __name__ == "__main__":
    check_columns()
