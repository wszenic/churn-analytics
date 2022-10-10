# library doc string
import logging
# import libraries
import os
from pathlib import Path
from typing import Any, Callable, Tuple

import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from config import CAT_COLUMNS, COLUMNS_TO_KEEP, RANDOM_FOREST_PARAMS_GRID, TARGET

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] | %(funcName)s:%(lineno)d | %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('./logs/log_churn.log'),
        logging.StreamHandler()
    ]
)


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    logging.info(f"Reading the input file from path {pth}")
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
        logging.info(f"Calling plotting function {plotting_function.__name__}")
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


def encoder_helper(df: pd.DataFrame, category_lst, response=None) -> pd.DataFrame:
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
        df[f"{col_name}_Churn"] = encode_one_target_column(df, col_name, 'Churn')

    return df


def encode_one_target_column(df: pd.DataFrame, grouped_column: str, averaged_column: str) -> pd.Series:
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


def perform_feature_engineering(df: pd.DataFrame, response=None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df = encoder_helper(df, CAT_COLUMNS)
    X, Y = split_x_and_y(df)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    logging.info("Creating classification reports")
    logging.info("Computing reports for RF model")
    train_rf_report = classification_report(y_train, y_train_preds_rf)
    test_rf_report = classification_report(y_test, y_test_preds_rf)

    print('random forest results')
    print('test results')
    print(test_rf_report)
    print('train results')
    print(train_rf_report)
    save_text_to_image(train_rf_report, './images/train_report_rf.png')
    save_text_to_image(test_rf_report, './images/test_report_rf.png')


    logging.info("Computing reports for LR model")
    train_lr_report = classification_report(y_train, y_train_preds_lr)
    test_lr_report = classification_report(y_test, y_test_preds_lr)

    print('logistic regression results')
    print('test results')
    print(test_lr_report)
    print('train results')
    print(train_lr_report)
    save_text_to_image(train_lr_report, './images/train_report_lr.png')
    save_text_to_image(test_lr_report, './images/test_report_lr.png')

    logging.info("Reports done")

def save_text_to_image(text: str, save_path: str) -> None:
    """
    Takes input text and converts it to png image and saves it
    Args:
        text: text to save
        save_path: directory with file name and extension

    Returns:
        None
    """
    logging.info("Saving text to image")
    report_image = Image.new('L', (200, 200))
    draw_image = ImageDraw.Draw(report_image)
    draw_image.text((20, 20), text, fill=(255,0,0))
    report_image.save(save_path)
    logging.info(f"Image saved to {save_path}")

def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    logging.info("Training Models:")

    logging.info("Training RF Classifier")
    rf_classifier = RandomForestClassifier(random_state=SEED)
    grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=RANDOM_FOREST_PARAMS_GRID)
    grid_search_rf.fit(X_train, y_train)

    logging.info("RF model trained, creating predictions")
    y_train_pred_rf = grid_search_rf.best_estimator_.predict(X_train)
    y_test_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
    logging.info("Predictions calculated with success")

    logging.info("Training Logistic Regression models")
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(X_train, y_train)

    logging.info("Logistic regression model trained, creating predictions")
    y_train_pred_lr = logistic_regression.predict(X_train)
    y_test_pred_lr = logistic_regression.predict(X_test)

    classification_report_image(y_train, y_test, y_train_pred_lr, y_train_pred_rf, y_test_pred_lr, y_test_pred_rf)


if __name__ == '__main__':
    df = import_data(r"./data/bank_data.csv")
    df = preprocess_columns(df)
    perform_eda(df)
    tran_test_sets = perform_feature_engineering(df)
    train_models(*tran_test_sets)
