import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def prepare_features_and_target(df):
    """
    Prepare features and target for model training
    
    Args:
        df: Preprocessed DataFrame with engineered features
        
    Returns:
        Tuple: X (features), y (target), feature_columns (list of feature names)
    """
    # Make sure we have the target column
    if 'churned' not in df.columns:
        raise ValueError("Target column 'churned' not found in DataFrame")
    
    # Identify features (exclude original categorical columns and target)
    exclude_cols = ['customer_id', 'churned', 'location', 'subscription_type', 
                    'payment_plan', 'payment_method', 'customer_service_inquiries']
    
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and target
    X = df[feature_columns]
    y = df['churned']
    
    return X, y, feature_columns

def train_model(X, y, model_type='xgboost'):
    """
    Train a churn prediction model
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model to train ('random_forest' or 'xgboost')
        
    Returns:
        dict: Trained model, feature importance, and model performance metrics
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = {
            'features': X.columns.tolist(),
            'importance': model.feature_importances_.tolist()
        }
        
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = {
            'features': X.columns.tolist(),
            'importance': model.feature_importances_.tolist()
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Return model, scaler, feature importance, and metrics
    return {
        'model': model,
        'scaler': scaler,
        'feature_importance': feature_importance,
        'metrics': metrics
    }

def save_model(model_data, model_path):
    """
    Save the trained model and associated data
    
    Args:
        model_data: Dictionary containing model and associated data
        model_path: Path to save the model
    """
    joblib.dump(model_data, model_path)

def train_and_save_model(df, model_path='models/churn_model.pkl', model_type='xgboost'):
    """
    End-to-end model training and saving
    
    Args:
        df: Preprocessed DataFrame with engineered features
        model_path: Path to save the model
        model_type: Type of model to train
        
    Returns:
        dict: Model data dictionary
    """
    # Prepare features and target
    X, y, feature_columns = prepare_features_and_target(df)
    
    # Train model
    model_data = train_model(X, y, model_type=model_type)
    
    # Add feature columns to model data
    model_data['feature_columns'] = feature_columns
    
    # Save model
    save_model(model_data, model_path)
    
    return model_data