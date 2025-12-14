import pandas as pd
import os
from typing import Tuple

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training and testing datasets from CSV files.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    
    Raises:
        FileNotFoundError: If either of the files does not exist.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at: {train_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing file not found at: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "house_prices_train.csv")
    test_file = os.path.join(base_dir, "house_prices_test.csv")
    
    try:
        train, test = load_data(train_file, test_file)
        print(f"Successfully loaded data.")
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        print("\nTrain Head:")
        print(train.head())
    except Exception as e:
        print(f"Error loading data: {e}")
