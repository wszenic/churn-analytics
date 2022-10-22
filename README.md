# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project produces the following models aimed to predict customer churn:
- Logistic Regression
- Random Forest

The project creates and saves the model along with a simple preprocessing and feature engineering.

During the execution, diagnostic data will be shown in the console and diagnostic plots will be saved 
to the `images` folder.

## Files and data description
The project has the following structre and contents:

```
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── config.py
├── data
│   └── bank_data.csv
├── images
├── logs
│   └── log_churn.log
├── models
├── requirements_py3.6.txt
├── requirements_py3.8.txt

└── utils.py

```

* `Guide.ipynb` is a guide to the project
* `README.md` is the file you are reading
* `churn_library.py` is the library with the functions to preprocess and train the model
* `churn_notebook.ipynb` is the notebook with the code to preprocess and train the model (same as `churn_library.py`,
this file was refactored to create the project)
* `models` is a folder with saved model files
* `data` is a folder with the data files (bank_data.csv)
* `images` is a folder with images generated during the modelling process
* `logs` is a folder with the log file
* `requirements_py3.6.txt` is a file with the requirements to run the project with python 3.6
* `requirements_py3.8.txt` is a file with the requirements to run the project with python 3.8
* `tests` is a folder with the tests (all tests are moved to this folder)
* `utils.py` is a file with some generic utilities used across the project
* `config.py` is a file with the configuration of the project

## Running Files
To run the project execute:

`python churn_library.py`

If You don't have the requirements installed, please run the following commands beforehand:
```
python venv venv
python -m pip install -r requirements3.8.txt
python activate venv
```




