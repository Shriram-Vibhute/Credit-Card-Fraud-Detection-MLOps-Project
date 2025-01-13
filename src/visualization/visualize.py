import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import joblib
import yaml
import seaborn as sns

from dvclive import Live
import mlflow
from mlflow.models import infer_signature

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

def evaluate(model, X, y, live, split, dvc_save_path, mlflow_save_path):
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

    # MLFlow metrics logging
    mlflow.log_metric("average_precision_score", avg_prec)
    mlflow.log_metric("roc_auc_score", roc_auc)

    # Logging Plots

    # DVC logging - ROC curve
    live.log_sklearn_plot(
        "roc", y, predictions, name = f"roc/{split}"
    )
    # MLFlow logging - ROC curve
    fpr, tpr, _ = metrics.roc_curve(y, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split}')
    plt.legend(loc="lower right")
    roc_plot_path = f"{mlflow_save_path}/{split}/roc_curve.png"
    pathlib.Path(f"{mlflow_save_path}/{split}").mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_plot_path)
    plt.close()
    mlflow.log_artifact(roc_plot_path)

    # DVC Logging - Precision Recall Curve
    live.log_sklearn_plot(
        "precision_recall", y, predictions, name=f"prc/{split}", drop_intermediate = True,
    )

    # MLFlow logging - Precision Recall Curve
    precision, recall, _ = metrics.precision_recall_curve(y, predictions)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {avg_prec:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {split}')
    plt.legend(loc="lower left")
    prc_plot_path = f"{mlflow_save_path}/{split}/prc_curve.png"

    plt.savefig(prc_plot_path)
    plt.close()
    mlflow.log_artifact(prc_plot_path)

    # DVC Logging - Confusion Matrix
    live.log_sklearn_plot(
        "confusion_matrix", y, prediction_by_class.argmax(-1), name=f"cm/{split}"
    )

    # MLFlow Logging - Confusion Matrix
    cm = confusion_matrix(y, prediction_by_class.argmax(-1), labels=model.classes_)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {split}')
    cm_plot_path = f"{mlflow_save_path}/{split}/confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()
    mlflow.log_artifact(cm_plot_path)

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

    # Output path - Where all the plots and metrics are going to be stored - dvclive
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Output path - MLFlow Artifacts
    mlflow_output_path = home_dir.as_posix() + '/mlexps'
    pathlib.Path(mlflow_output_path).mkdir(parents=True, exist_ok=True)

    # MLFlow model saving path
    mlflow_model_output_path = home_dir.as_posix() + '/mlflow_model'
    pathlib.Path(mlflow_model_output_path).mkdir(parents=True, exist_ok=True)

    # Loading data and model
    X_train, X_test, y_train, y_test = load_data(data_path = input_path)
    model = load_model(model_path = model_path)

    # Features list
    feature_names = X_train.columns.to_list()

    # Parameters path
    params_path = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_path, mode = 'r'))
    params_dict = {}
    for topic in params:
        for each_param in params[topic]:
            params_dict[f"{topic}_{each_param}"] = params[topic][each_param]
    
    # MLFlow setup
    mlflow.set_tracking_uri(uri="http://localhost:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("RandomForests Experiments")

    # Evaluation
    with Live(output_path, dvcyaml = False) as live:
        with mlflow.start_run():
            # Logging Parameters
            mlflow.log_params(params_dict)

            # Train
            evaluate(
                model = model,
                X = X_train,
                y = y_train,
                live = live,
                split = "train",
                dvc_save_path = output_path,
                mlflow_save_path = mlflow_output_path,
            )

            # Test
            evaluate(
                model = model,
                X = X_test,
                y = y_test,
                live = live,
                split = "test",
                dvc_save_path = output_path,
                mlflow_save_path = mlflow_output_path,
            )

            # Saving important plots - feature_importance_plots
            save_plots(live, model, feature_names)

        # Logging Model
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            signature=signature,
            input_example=X_train,
            registered_model_name="RandomForest Classifier",
        )

if __name__ == "__main__":
    main()