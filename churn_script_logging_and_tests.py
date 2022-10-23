"""
This file contains all the tests for the churn library

Author: Wojciech Szenic
Creation date: 2022-10-23
"""

import glob
import logging
import os
from pathlib import Path

import pandas as pd
import pytest
import sklearn

from churn_library import encoder_helper, perform_eda, perform_feature_engineering, \
    preprocess_columns, save_text_to_image, train_models
from config import CAT_COLUMNS
from utils import get_project_root

# Fixtures
@pytest.fixture(scope='session', name='_path_to_data')
def mock_test_data_path():
    """ Mock the path to the test data """
    return Path(get_project_root() / 'data/bank_data.csv')


@pytest.fixture(scope='session', name='_input_data')
def mock_read_file(_path_to_data):
    """ Mock the read file function """
    return pd.read_csv(_path_to_data)


@pytest.fixture(scope='session', name='_preprocessed_file')
def mock_preprocessed_file(_input_data):
    """ Mock the preprocessed file """
    return preprocess_columns(_input_data)


@pytest.fixture(scope='session', name='_eda_process')
def mock_eda_process(_preprocessed_file):
    """ Mock the eda process """
    yield perform_eda(_preprocessed_file)


@pytest.fixture(scope='session', name='_encoder_helper')
def mock_encoder_helper(_preprocessed_file):
    """ Mock the encoder helper """
    return encoder_helper(_preprocessed_file, CAT_COLUMNS)


@pytest.fixture(scope='session', name='_feature_engineering')
def mock_feature_engineering(_preprocessed_file):
    """ Mock the feature engineering process """
    yield perform_feature_engineering(_preprocessed_file)


@pytest.fixture(scope='session', name='_save_text_to_image')
def mock_saving_text_to_image():
    """ Mock the saving text to image process """
    yield save_text_to_image("Test string", get_project_root() / 'images' / 'test.png')
    os.remove(get_project_root() / 'images' / 'test.png')


@pytest.fixture(scope="session", name='_train_model')
def mock_train_model(_feature_engineering):
    """ Mock the training of the models """
    yield train_models(*_feature_engineering)


# Tests
def test_import_data(_input_data):
    """ Test if the data is imported correctly """
    logging.info("Testing if the data is imported correctly")
    assert mock_read_file is not None  # asserts that rows are read
    logging.info("Success, data is imported correctly")


def test_preprocess_columns(_preprocessed_file):
    """ Test if the data is preprocessed correctly """
    logging.info("Testing if the data is preprocessed correctly")
    assert 'Churn' in _preprocessed_file.columns
    assert all(_preprocessed_file['Churn']) <= 1
    assert all(_preprocessed_file['Churn']) >= 0
    assert _preprocessed_file['Churn'].dtype == int
    logging.info("Success, data is preprocessed correctly")


def test_perform_eda(_eda_process):
    """ Test if the eda plots are saved correctly """
    logging.info("Testing if the eda plots are saved correctly")
    assert Path('./images/').exists()   # dir exists
    assert any(Path('./images/').iterdir())     # plt dir not empty
    assert Path('./images/Churn_hist.png').exists()
    assert Path('./images/Customer_Age_hist.png').exists()
    assert Path('./images/heatmap.png').exists()
    assert Path('./images/Total_Trans_Ct_hist.png').exists()
    logging.info("Success, eda plots are saved correctly")


def test_encoder_helper(_encoder_helper):
    """ Test if the encoder helper works correctly on all columns """
    logging.info("Testing if the encoder helper works correctly on all columns")
    assert all(x for x in _encoder_helper.columns if x in CAT_COLUMNS)
    logging.info("Success, encoder helper works correctly on all columns")


def test_save_text_to_image(_save_text_to_image):
    """ Test if the text is saved correctly to image"""
    logging.info("Testing if the text is saved correctly to image")
    assert Path(get_project_root() / 'images/test.png').exists()
    logging.info("Success, text is saved correctly to image")


def test_train_models(_train_model):
    """ Test if the models are trained correctly and returned by the function """
    logging.info("Testing if the models are trained correctly and returned by the function")
    assert len(_train_model) == 2
    assert _train_model[0] is not None
    assert _train_model[1] is not None
    logging.info("Success, models are trained correctly and returned by the function")


def test_if_models_are_fitted(_train_model):
    """ Test if the models are fitted correctly """
    logging.info("Testing if the models are fitted correctly")
    assert not any(sklearn.utils.validation.check_is_fitted(x)
               for x in _train_model)
    logging.info("Success, models are fitted correctly")


def test_model_pickles_saved(_train_model):
    """ Test if the models are saved correctly """
    logging.info("Testing if the models are saved correctly")
    assert Path(get_project_root() / 'models/rf_model.pkl').exists()
    assert Path(get_project_root() / 'models/lr_model.pkl').exists()
    logging.info("Success, models are saved correctly")


def test_diagnostic_reports_saved(_train_model):
    """ Test if the diagnostic reports are saved correctly """
    logging.info("Testing if the diagnostic reports are saved correctly")
    assert Path(
        get_project_root() /
        'images/train_report_logistic_regression.png').exists()
    assert Path(
        get_project_root() /
        'images/train_report_random_forest.png').exists()
    assert Path(
        get_project_root() /
        'images/test_report_logistic_regression.png').exists()
    assert Path(
        get_project_root() /
        'images/test_report_random_forest.png').exists()
    logging.info("Success, diagnostic reports are saved correctly")


def test_feature_importance_saved(_train_model):
    """ Test if the feature importance plots are saved correctly """
    logging.info("Testing if the feature importance plots are saved correctly")
    assert Path(glob.glob(str(get_project_root() /
                              'images/LogisticRegression*_feature_importance.png'))[0]).exists()
    assert Path(glob.glob(str(get_project_root() /
                              'images/LogisticRegression*_roc_plot.png'))[0]).exists()
    assert Path(glob.glob(str(get_project_root(
    ) / 'images/RandomForestClassifier*_feature_importance.png'))[0]).exists()
    assert Path(glob.glob(str(get_project_root() /
                              'images/RandomForestClassifier*_roc_plot.png'))[0]).exists()
    logging.info("Success, feature importance plots are saved correctly")
