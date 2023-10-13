# Sales Prediction.

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

### 3. Install the required libraries 
```
pip install -r requirements.txt
```

## Directory Structure

├── artifacts
├── catboost_info
├── input
├── notebooks
│   └── exploration.ipynb
├── pipelines
├── src
│   ├── components
│   └── pipeline
│       ├── __init__.py
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
├── templates
├── tests
│   ├── test_data_encoding.py
│   ├── test_data_imputation.py
│   ├── test_data_transformation.py
│   ├── test_extract_features.py
│   └── test_feature_selection.py
├── .gitignore
├── README.md
├── app.py
├── best_params.json
├── config.py
├── pyproject.toml
├── requirements.txt
└── setup.py

## [Usage]
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
-pandas
-numpy
-seaborn
-matplotlib
-scikit-learn
-dill

Development Dependencies:
-pytest
-black
-flake8

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


```bash
# Check Python installation
python --version


