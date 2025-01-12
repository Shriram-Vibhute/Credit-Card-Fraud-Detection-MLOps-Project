import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import joblib

from dvclive import Live

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay

def load_data(data_path: str):
    # Loading Data
    train_df = pd.read_csv(data_path + "/train.csv")
    test_df = pd.read_csv(data_path + "/test.csv")

    # Splitting
    X_train = train_df.drop(columns = ['Class'])
    X_test = test_df.drop(columns = ['Class'])
    y_train = train_df['Class']
    y_test = test_df['Class']

    return X_train, X_test, y_train, y_test

def load_model(model_path: str):
    # Loading Model
    model = joblib.load(model_path + "/model.joblib")
    return model

def evaluate(model, X, y, live, split, save_path):
    # Predictions
    prediction_by_class = model.predict_proba(X)
    predictions = prediction_by_class[:, 1] # Fetching the probability of fraud(Yes) credit cards

    # Evaluation Metrics
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)

    # DVC Logging
    if not live.summary:
        live.summary = {
            "avg_prec": {},
            "roc_auc": {}
        }
    live.summary['avg_prec'][split] = avg_prec
    live.summary['roc_auc'][split] = roc_auc

    # Logging Plots
    live.log_sklearn_plot(
        "roc", y, predictions, name = f"roc/{split}"
    )
    live.log_sklearn_plot(
        "precision_recall", y, predictions, name=f"prc/{split}", drop_intermediate = True,
        # Threshold: The probability value above which a sample is classified as the positive class. For example, if the threshold is 0.95, only samples with predicted probabilities greater than or equal to 0.95 are classified as positive. (in test.json)
    )
    live.log_sklearn_plot(
        "confusion_matrix", y, prediction_by_class.argmax(-1), name=f"cm/{split}"
    )

def save_plots(live, model, feature_names):
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)

def main():
    # Creating Paths
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent

    # Data paths
    input_path = home_dir.as_posix() + "/data/processed"

    # Model path
    model_path = home_dir.as_posix() + "/models"

    # Output path - Where all the plots and matrics are going to be sotred - dvclive
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Loading data and model
    X_train, X_test, y_train, y_test = load_data(data_path = input_path)
    model = load_model(model_path = model_path)

    # Features list
    feature_names = X_train.columns.to_list()

    # Evaluation
    with Live(output_path, dvcyaml = False) as live:
        # Train
        evaluate(
            model = model,
            X = X_train,
            y = y_train,
            live = live,
            split = "train",
            save_path = output_path
        )

        # Test
        evaluate(
            model = model,
            X = X_test,
            y = y_test,
            live = live,
            split = "test",
            save_path = output_path
        )

        # Saving important plots - feature_importance_plots
        save_plots(live, model, feature_names)

if __name__ == "__main__":
    main()