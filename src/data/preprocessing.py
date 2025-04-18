import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def fix_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame with missing values
        
    Returns:
        DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    # Fill missing values for numerical columns
    numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df_copy[col].isnull().sum() > 0:
            # Use random sampling from non-null values
            df_copy[col] = df_copy[col].apply(
                lambda x: np.random.choice(df_copy[col].dropna()) if pd.isnull(x) else x
            )
    
    # Fill missing values for categorical columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_copy[col].isnull().sum() > 0:
            # Use random sampling from non-null values
            df_copy[col] = df_copy[col].apply(
                lambda x: np.random.choice(df_copy[col].dropna()) if pd.isnull(x) else x
            )
    
    return df_copy

def encode_categorical_features(df):
    """
    Encode categorical features in the dataset
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame: DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    # Encode location based on frequency
    if 'location' in df_encoded.columns:
        location_counts = df_encoded['location'].value_counts()
        location_mapping = location_counts.to_dict()
        df_encoded['location_encoded'] = df_encoded['location'].map(location_mapping)
    
    # One-hot encode subscription_type
    if 'subscription_type' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['subscription_type'], prefix='subscription')
    
    # Map payment plan to binary
    if 'payment_plan' in df_encoded.columns:
        df_encoded['payment_plan_encoded'] = df_encoded['payment_plan'].map({'Monthly': 0, 'Yearly': 1})
    
    # One-hot encode payment_method
    if 'payment_method' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['payment_method'], prefix='payment')
    
    # Map customer service inquiries to ordinal values
    if 'customer_service_inquiries' in df_encoded.columns:
        inquiry_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df_encoded['customer_service_inquiries_encoded'] = df_encoded['customer_service_inquiries'].map(inquiry_mapping)
    
    return df_encoded

def preprocess_data(df):
    """
    Full preprocessing pipeline
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame: Preprocessed DataFrame
    """
    # Fix missing values
    df_clean = fix_missing_values(df)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df_clean)
    
    return df_encoded