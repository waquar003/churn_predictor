# Setup Instructions

## To start, run the following commands:
```sh
python -m venv churn_env
churn_env\Scripts\activate
pip install -r requirements.txt
```

## Project Structure
```
churn_predictor/
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data ready for modeling
│
├── notebooks/                 # Jupyter notebooks for exploration and development
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Data cleaning and preparation
│   │   └── features.py        # Feature engineering
│   │
│   ├── models/                # Model-related code
│   │   ├── __init__.py
│   │   ├── train.py           # Model training
│   │   └── predict.py         # Prediction functions
│   │
│   └── visualization/         # Visualization utilities
│       ├── __init__.py
│       └── plots.py           # Functions for creating plots
│
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── app/                       # Deployment application
│   ├── __init__.py
│   ├── api.py                 # API endpoints
│   ├── static/                # Static files (CSS, JS)
│   └── templates/             # HTML templates
│
├── models/                    # Saved model files
│   └── churn_model_v1.pkl
│
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup file
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

