# This script is responsible for creating train and test datasets.
import pathlib
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str):
     # Load and return the dataset from the specified CSV file.
    return pd.read_csv(file_path)

def split_data(df, params):
    # Split the dataset into training and testing sets based on the provided parameters.
    train_df, test_df = train_test_split(df, test_size = params['test_split'], random_state = params['seed'])
    return train_df, test_df

def save_data(train_df, test_df, save_path) -> None:
    # Create the directory if it doesn't exist and save the train and test datasets as CSV files.
    pathlib.Path(save_path).mkdir(parents = True, exist_ok = True)
    train_df.to_csv(save_path + '/train.csv', index = False)
    test_df.to_csv(save_path + '/test.csv', index = False)

def main():
    # Creating paths
    current_path = pathlib.Path(__file__).resolve()
    home_dir = current_path.parent.parent.parent

    # Parameter paths
    parameters_path = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(parameters_path, mode = 'r'))['make_dataset']

    # Data paths
    input_path = home_dir.as_posix() + "/data/raw/creditcard.csv"
    output_path = home_dir.as_posix() + "/data/processed"

    # Loading data
    df = load_data(input_path) # This function return data

    # Splitting data
    train, test = split_data(df, params)

    # Saving the data
    save_data(train, test, save_path = output_path)

if __name__ == "__main__":
    main() # Running a script as module