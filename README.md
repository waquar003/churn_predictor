# Customer Churn Predictor

A machine learning-based **Customer Churn Predictor** application with an intuitive **Streamlit** interface. This project allows users to upload their own datasets, preprocess the data, generate predictions, and explore insights and trends within customer data.

---

## 🚀 Features

- 📁 **Upload Your Dataset**: Easily upload a CSV file containing customer data.
- 🧼 **Data Preprocessing**: Clean and prepare data with built-in preprocessing tools.
- 🧠 **Model Prediction**: Generate churn predictions using a trained machine learning model.
- 📊 **Insight Generation**: Visualize trends, patterns, and key insights from the uploaded data.
- 🌐 **Streamlit Web Interface**: User-friendly, interactive web UI for a seamless experience.

---

## 🗂️ Project Structure

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
│   ├── data/                  # Data processing modules
│   │   ├── preprocessing.py   # Data cleaning and preparation
│   │   └── features.py        # Feature engineering
│   │
│   ├── models/                # Model-related code
│   │   ├── train.py           # Model training
│   │   └── predict.py         # Prediction functions
│   │
│   └── visualization/         # Visualization utilities
│       └── plots.py           # Functions for creating plots
│
├── generate_model.py          # Pipeline for training the model
├── app.py                     # Streamlit app for deployment
├── models/                    # Saved model files
│   └── churn_model_v1.pkl
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup file
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

---

## 🛠️ Setup Instructions

To get started, clone the repo and run the following commands:

```sh
python -m venv churn_env
churn_env\Scripts\activate
pip install -r requirements.txt
```

Then, launch the Streamlit app:

```sh
streamlit run app.py
```

---

## 📈 Technologies Used

- **Python**
- **Pandas**, **NumPy**, **scikit-learn**
- **Streamlit**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebook**

---

## 📌 Use Case

This tool is especially useful for:

- Customer retention teams
- Marketing analysts
- Data scientists exploring churn behavior
- Businesses looking to understand customer attrition trends

---
