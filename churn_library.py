# library doc string
import logging
# import libraries
import os
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

COLUMNS_TO_KEEP = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

TARGET = 'Churn'

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
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

    def wrapper(*args, **kwargs):
        logging.info(f"Calling plotting function {plotting_function.__name__}")
        return plotting_function(*args, **kwargs)

    plt.clf()
    return wrapper


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
            response: string of response name [optional argument that could be used for naming variables or index y column]

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
    pass


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
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == '__main__':
    df = import_data(r"./data/bank_data.csv")
    df = preprocess_columns(df)
    perform_eda(df)
    df = encoder_helper(df, cat_columns)
    print(df.head())
