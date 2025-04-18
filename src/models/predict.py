import pandas as pd
import numpy as np
import joblib

def load_model(model_path):
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        dict: Model data dictionary
    """
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_churn(df, model_data=None, model_path=None):
    """
    Make churn predictions for customers
    
    Args:
        df: DataFrame with features
        model_data: Model data dictionary (optional)
        model_path: Path to the saved model (required if model_data is None)
        
    Returns:
        DataFrame: Original DataFrame with churn predictions and probabilities
    """
    if model_data is None and model_path is None:
        raise ValueError("Either model_data or model_path must be provided")
    
    if model_data is None:
        model_data = load_model(model_path)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Make sure we have all required features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    # Extract features
    X = df[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    df_result = df.copy()
    df_result['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
    df_result['predicted_churn'] = model.predict(X_scaled)
    
    # Add churn risk category
    df_result['churn_risk_category'] = pd.cut(
        df_result['churn_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Moderate', 'High']
    )
    
    return df_result

def get_churn_factors(df, customer_id, model_data, top_n=5):
    """
    Get top factors contributing to churn for a specific customer

    Args:
        df: DataFrame with predictions
        customer_id: ID of the customer
        model_data: Model data dictionary
        top_n: Number of top factors to return

    Returns:
        dict: Top factors contributing to churn
    """
    # Get feature importance from model
    feature_importance = model_data['feature_importance']
    features = feature_importance['features']
    importance = feature_importance['importance']

    # Get customer data
    customer_data = df[df['customer_id'] == customer_id]
    if customer_data.empty:
        raise ValueError(f"Customer with ID {customer_id} not found")

    # Calculate feature contribution based on importance and feature value
    contributions = []
    for feature, importance_value in zip(features, importance):
        if feature in customer_data.columns:
            # Get feature value and its percentile rank in the dataset
            value = customer_data[feature].values[0]

            # Skip if feature is all zeros or has only one unique value
            if df[feature].nunique() <= 1:
                continue

            percentile = (df[feature] <= value).mean()

            # Calculate concern level
            if feature in ['pause_frequency', 'song_skip_rate', 'churn_risk_score',
                           'customer_service_inquiries_encoded']:
                concern_level = percentile
            else:
                concern_level = 1 - percentile

            contribution = importance_value * concern_level

            contributions.append({
                'feature': feature,
                'value': value,
                'percentile': percentile,
                'importance': importance_value,
                'contribution': contribution,
            })

    # Sort by contribution and get top N
    contributions.sort(key=lambda x: x['contribution'], reverse=True)
    top_factors = contributions[:top_n]

    return {
        'customer_id': customer_id,
        'churn_probability': customer_data['churn_probability'].values[0],
        'churn_risk_category': customer_data['churn_risk_category'].values[0],
        'top_factors': top_factors
    }
