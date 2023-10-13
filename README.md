# Sales Prediction

## UI Interface:

![Home page interface](https://github.com/KonstantinosTsoumas/sales-prediction/tree/main/images/input_page.png)

![UI input page](https://github.com/KonstantinosTsoumas/sales-prediction/tree/main/images/home.png)


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Authors](#authors)
- [Modify and Extend](#modify-and-extend)
- [License](#license)
- [Contributing](#contributing)

## Getting Started

## Introduction

This project is primarily aimed at predicting the value in sales of a product as trained on a Supply Chain Dataset.
You can find more about the dataset <a href="https://data.mendeley.com/datasets/8gx2fvg2k6/5">here</a>.


### Prerequisites

- Python 3.7 or higher
- Poetry package manager (optional)

### Installation
### 1. Clone the repository (please check copy the repository url from GitHub):
```
git clone <repository-url>
```

### 2. Navigate to project (if not already)
```
cd sales-prediction
```

### 3. Install the required libraries using setup
```
pip install -e .
```

## Directory Structure

```
├── artifacts                # Storing build artifacts or temporary files
├── catboost_info            # Information and logs related to CatBoost model
├── input                    # Folder for input data files
├── notebooks                # Jupyter notebooks
│   └── exploration.ipynb    # Notebook for data exploration
├── pipelines                # Data pipelines for ETL tasks
├── src                      # Source code directory
│   ├── components           # Reusable code components/modules
│   └── pipeline             # Core pipeline code
│       ├── __init__.py      # Initialize pipeline package
│       ├── exception.py     # Custom exceptions for the pipeline
│       ├── logger.py        # Logging utility
│       └── utils.py         # Miscellaneous utility functions
├── templates                # Template files (if needed)
├── tests                    # Test scripts and files
│   ├── test_data_encoding.py        # Tests for data encoding
│   ├── test_data_imputation.py      # Tests for data imputation
│   ├── test_data_transformation.py  # Tests for data transformations
│   ├── test_extract_features.py     # Tests for feature extraction
│   └── test_feature_selection.py    # Tests for feature selection
├── .gitignore               # Git ignore file
├── README.md                # Project readme
├── app.py                   # Main application file
├── best_params.json         # JSON file to store best parameters for models
├── config.py                # Configuration file
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Required packages
└── setup.py                 # Project setup script
```


## Usage
There are two options. 
You can either run the whole code from scratch locally or you can build the Flask app and use the pre-trained model.

Run from scratch option:
### 1. Clone the repository (please check copy the repository url from GitHub):
```
git clone <repository-url>
```

### 2. Navigate to project (if not already)
```
cd sales-prediction
```

### 3. Install the requirements
```
pip install -r requirements.txt
```

## Dependencies
You can also use Poetry for dependency management.

Main Dependencies:
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* dill

Development Dependencies:
* pytest
* black
flake8

Run the following command to install project dependencies:
```
poetry install
```

## Authors

[Konstantinos Tsoumas](https://github.com/KonstantinosTsoumas)

## Modify and extend

Feel free to modify the code and adapt it to your specific needs.
Add additional tests in the tests directory to ensure the correctness of the code.
Update the documentation and README.md file to reflect any changes made.

## License

This project is licensed under the MIT License. Feel free to use and modify the code according to the terms of the license.

## Contributing

Contributions to this project are welcome. Feel free to open issues and submit pull requests to suggest improvements, add new features, or fix any bugs.

