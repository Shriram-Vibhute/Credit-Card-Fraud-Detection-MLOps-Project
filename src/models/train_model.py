# This script is responsible for training and saving the model.
import numpy as np
import pandas as pd
import pathlib
import yaml

from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(data_path):
    train_df = pd.read_csv(data_path + "/train.csv")

    # Splitting input and output data
    y = train_df['Class']
    X = train_df.drop(columns = ['Class'])
    return X, y

def train_model(X_train, y_train, params):
    # Creating a model
    model = RandomForestClassifier(
        n_estimators = params['n_estimators'],
        max_depth = params['max_depth'],
        random_state = params['seed']
    )

    # Training a model
    model.fit(X_train, y_train)
    
    # Returning a model
    return model

def save_model(model, save_path):
    pathlib.Path(save_path).mkdir(parents = True, exist_ok = True)

    # Saving the model into models folder
    joblib.dump(model, save_path + '/model.joblib')

def main() -> None:
    # Creating paths
    current_path = pathlib.Path(__file__).resolve()
    home_dir = current_path.parent.parent.parent

    # Parameter paths
    parameters_path = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(parameters_path, mode = 'r'))['train_model']

    # Data paths
    output_path = home_dir.as_posix() + "/data/processed"

    # Model path
    model_path = home_dir.as_posix() + "/models"

    # Load training data
    X, y = load_data(output_path)

    # Model Training
    model = train_model(X_train = X, y_train = y, params = params)

    # Saving the model
    save_model(model, model_path)

if __name__ == "__main__":
    main()
