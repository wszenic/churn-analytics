"""
Module used to create a churn model and save related files to disc (diagnostics, model, etc)

Author: Wojciech Szenic
Creation date: 2022-10-23
"""
import logging
import os
from collections import namedtuple
from pathlib import Path
from typing import Callable, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from config import CAT_COLUMNS, COLUMNS_TO_KEEP, RANDOM_FOREST_PARAMS_GRID, TARGET
from utils import get_project_root

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] | %(funcName)s:%(lineno)d | %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(
            get_project_root() /
            'logs/log_churn.log'),
        logging.StreamHandler()])

ModelDatasets = namedtuple(
    'model_datasets', [
        'x_train', 'x_test', 'y_train', 'y_test'])

PredcitionResults = namedtuple(
    'prediction_results', [
        "model_name", "train_pred", "test_pred"])


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    logging.info("Reading the input file from path %s", pth)
    df = pd.read_csv(pth)
    logging.info("Success, file read")
    return df


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    perform basic preprocessing before any analysis can be done.
    The sequence is as follows:
        - create Churn column [int] - 1 if churned 0 if didn

    inputs:
            df: pandas dataframe
    outputs
            df: pandas dataframe (with added columns
    """
    df['Churn'] = (df['Attrition_Flag'] != 'Existing Customer').astype(int)

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    __calculate_basic_descriptive_statistics(df)
    logging.info("Saving graphs to images folder...")
    logging.info("Creating directory if it doesnt exist")
    Path('./images/').mkdir(exist_ok=True)
    logging.info("Creating plots...")
    create_histogram_from_column(df, 'Churn')
    create_histogram_from_column(df, 'Customer_Age')
    create_counts_plot(df, 'Total_Trans_Ct')
    create_heatmap(df)


def plot_settings(plotting_function: Callable) -> Callable:
    """
    decorator to pass unified settings into a plotting function:
     - the same plot size
     - clear 'current figure' to avoid bugs due to matplotlib's global state

    inputs:
            plotting_function: a function, producing a matplotlib graph
    outputs:
            None
    """
    plt.figure(figsize=(20, 10))

    def plotting_fn(*args, **kwargs):
        logging.info(
            "Calling plotting function %s",
            plotting_function.__name__)
        return plotting_function(*args, **kwargs)

    plt.clf()
    return plotting_fn


@plot_settings
def create_histogram_from_column(df: pd.DataFrame, column_name: str) -> None:
    """
    Creates a histogram of fixed dimensions and saves it to the './images' directory
    input:
            df: pandas dataframe
            column_name: string name of a column present in the dateframe in the first arg
    output:
            None
    """
    df[column_name].hist()
    plt.savefig(f'./images/{column_name}_hist.png')
    plt.clf()


@plot_settings
def create_counts_plot(df: pd.DataFrame, column_name: str) -> None:
    """
    Creates a seaborn histogram of fixed dimensions and saves it to the './images' directory
    input:
            df: pandas dataframe
            column_name: string name of a column present in the dateframe in the first arg
    output:
            None
    """
    sns.distplot(df[column_name], kde=True)
    plt.savefig(f'./images/{column_name}_hist.png')


@plot_settings
def create_heatmap(df: pd.DataFrame) -> None:
    """
    Creates a seaborn heatmap and saves it to the './images' directory
    input:
            df: pandas dataframe
            column_name: string name of a column present in the dateframe in the first arg
    output:
            None
    """
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/heatmap.png')


def __calculate_basic_descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Computes and prints to the console:
     - shape of the dataframe
     - count of nulls per column
     - summary statistics

    input:
            df: pandas dataframe

    output:
            None
    """
    logging.info("Computing basic descriptive statistics...")
    print("== Starting EDA ==")
    print(f"Shape of the df is: {df.shape}")
    print('-' * 30)
    print(f"Sum of nulls per column in the df: \n{df.isnull().sum()}")
    print('-' * 30)
    print(f"Summary statistics for the df: {df.describe()}")


def encoder_helper(
        df: pd.DataFrame,
        category_lst) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
                variables or index y column]

    output:
            df: pandas dataframe with new columns for categorical variables
    """
    for col_name in category_lst:
        df[f"{col_name}_Churn"] = encode_one_target_column(
            df, col_name, 'Churn')

    return df


def encode_one_target_column(
        df: pd.DataFrame,
        grouped_column: str,
        averaged_column: str) -> pd.Series:
    """
    perform encoding over one categorical variable and compute mean of another column

    Args:
        df: pandas dataframe
        grouped_column: column over which the data is going to be grouped
        averaged_column: column for which the average is computed

    Returns:
        pd.Series of averaged values

    """
    return df.groupby(grouped_column)[averaged_column].transform('mean')


def split_x_and_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits one modelling dataframe into set of X and Y
    Args:
        df: dataframe to split

    Returns:
        tuple of two dataframes, X and Y

    """
    return df[COLUMNS_TO_KEEP], df[TARGET]


def perform_feature_engineering(df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df = encoder_helper(df, CAT_COLUMNS)
    x, y = split_x_and_y(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    return ModelDatasets(x_train, x_test, y_train, y_test)


def classification_report_image(y_train: pd.DataFrame,
                                y_test: pd.DataFrame,
                                lr_results: PredcitionResults,
                                rf_results: PredcitionResults) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            lr_results: logistic regression results
            rf_results: random forest results

    output:
             None
    """
    logging.info("Creating classification reports")

    for model_results in [lr_results, rf_results]:
        __create_single_classification_report(y_train, y_test, model_results)

    logging.info("Reports done")


def __create_single_classification_report(
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        pred_results: PredcitionResults) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            lr_results: logistic regression results
            rf_results: random forest results

    output:
             None
    """
    logging.info("Creating classification reports")
    logging.info("Computing reports for %s", pred_results.model_name)
    train_report = classification_report(y_train, pred_results.train_pred)
    test_report = classification_report(y_test, pred_results.test_pred)

    print('% results', pred_results.model_name)
    print('test results')
    print(test_report)
    print('train results')
    print(train_report)
    save_text_to_image(
        train_report,
        f'./images/train_report_{pred_results.model_name}.png')
    save_text_to_image(
        test_report,
        f'./images/test_report_{pred_results.model_name}.png')


def save_text_to_image(text: Union[str, Path], save_path: str) -> None:
    """
    Takes input text and converts it to png image and saves it
    Args:
        text: text to save
        save_path: directory with file name and extension. Both str and Path are accepted

    Returns:
        None
    """
    logging.info("Saving text to image")
    report_image = Image.new('L', (512, 512))
    draw_image = ImageDraw.Draw(report_image)
    draw_image.text((10, 10), text, fill=(255))
    report_image.save(save_path)
    logging.info("Image saved to %s", save_path)


def feature_importance_plot(model, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    logging.info("Calculating feature importances")

    feat_importance = __get_feature_importances(model)

    # Add bars
    plt.clf()
    sns.barplot(x=feat_importance.feature_names, y=feat_importance.importance)
    plt.xticks(
        np.arange(
            0,
            feat_importance.shape[0]),
        feat_importance.feature_names,
        rotation=90)
    # Add feature names as x-axis labels
    logging.info("Success, feature importance plot created")

    logging.info("Saving feature importance plot")
    plt.savefig(output_pth)


def __get_feature_importances(model) -> pd.DataFrame:
    """
    Returns feature importances for a given model
    Args:
        model: model to get feature importances from

    Returns:
        dataframe with feature importances

    """
    logging.info("Getting feature importances")

    importance_lookup_dict = {
        "<class 'sklearn.linear_model._logistic.LogisticRegression'>": "coef_",
        "<class 'sklearn.ensemble._forest.RandomForestClassifier'>": "feature_importances_"}

    model_type = str(type(model))
    importance_vector = getattr(model, importance_lookup_dict[model_type])

    feat_importance = pd.DataFrame({
        "importance": importance_vector.flatten(),
        "feature_names": model.feature_names_in_
    })
    feat_importance.sort_values('importance', ascending=False, inplace=True)
    return feat_importance


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    logging.info("Training Models:")

    logging.info("Training RF Classifier")
    rf_classifier = RandomForestClassifier(random_state=SEED)
    grid_search_rf = GridSearchCV(
        estimator=rf_classifier,
        param_grid=RANDOM_FOREST_PARAMS_GRID)
    grid_search_rf.fit(x_train, y_train)

    logging.info("RF model trained, creating predictions")
    y_train_pred_rf = grid_search_rf.best_estimator_.predict(x_train)
    y_test_pred_rf = grid_search_rf.best_estimator_.predict(x_test)
    rf_results = PredcitionResults(
        "random_forest",
        y_train_pred_rf,
        y_test_pred_rf)
    logging.info("Predictions calculated with success")

    logging.info("Training Logistic Regression models")
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(x_train, y_train)

    logging.info("Logistic regression model trained, creating predictions")
    y_train_pred_lr = logistic_regression.predict(x_train)
    y_test_pred_lr = logistic_regression.predict(x_test)
    lr_results = PredcitionResults(
        "logistic_regression",
        y_train_pred_lr,
        y_test_pred_lr)
    logging.info("Predictions calculated with success")

    logging.info("Creating classification reports")
    classification_report_image(
        y_train,
        y_test,
        lr_results,
        rf_results
    )

    create_roc_plot(logistic_regression, x_test, y_test)
    create_roc_plot(grid_search_rf.best_estimator_, x_test, y_test)
    logging.info("ROC curve plot created")

    logging.info("Saving models")
    joblib.dump(
        grid_search_rf.best_estimator_,
        get_project_root() /
        'models/rf_model.pkl')
    joblib.dump(
        logistic_regression,
        get_project_root() /
        'models/lr_model.pkl')
    logging.info("Models saved")

    logging.info("Training done")

    return grid_search_rf.best_estimator_, logistic_regression


@plot_settings
def create_roc_plot(model, x_test, y_test):
    """
    creates and stores the ROC curve for the model
    input:
            model: model object
            x_test: pandas dataframe of X values
            y_test: pandas series of y values

    output:
             None
    """
    logging.info("Creating ROC curve")
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.savefig(f'./images/{str(model)}_roc_plot.png')


def main():
    """
    main functions, runs churn modelling logic
    Returns:

    """
    logging.info("Starting the churn modelling process")
    df = import_data(r"./data/bank_data.csv")
    df = preprocess_columns(df)
    perform_eda(df)

    tran_test_sets = perform_feature_engineering(df)
    models = train_models(*tran_test_sets)

    for model in models:
        feature_importance_plot(
            model, f'./images/{str(model)}_feature_importance.png')

    logging.info("Churn modelling process finished")


if __name__ == '__main__':
    main()
