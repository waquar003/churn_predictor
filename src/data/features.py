import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Engineer features for churn prediction
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    df_engineered = df.copy()
    
    # Calculate tenure days from signup date (assuming signup_date is days from a reference point)
    if 'signup_date' in df_engineered.columns:
        df_engineered['tenure_days'] = abs(df_engineered['signup_date'])
        # Normalized Signup recency feature
        df_engineered['signup_recency'] = df_engineered['tenure_days'] / df_engineered['tenure_days'].max()
        # Drop the original signup_date column
        df_engineered.drop(columns=['signup_date'], inplace=True)
    
    # Calculate engagement score (songs played per hour)
    if 'weekly_songs_played' in df_engineered.columns and 'weekly_hours' in df_engineered.columns:
        df_engineered['engagement_score'] = df_engineered.apply(
            lambda row: row['weekly_songs_played'] / row['weekly_hours'] if row['weekly_hours'] > 0 else 0, 
            axis=1
        )
    
    # Calculate session frequency (sessions per week)
    if 'weekly_hours' in df_engineered.columns and 'average_session_length' in df_engineered.columns:
        df_engineered['session_frequency'] = df_engineered.apply(
            lambda row: row['weekly_hours'] / row['average_session_length'] if row['average_session_length'] > 0 else 0,
            axis=1
        )
    
    # Calculate song diversity ratio
    if 'weekly_unique_songs' in df_engineered.columns and 'weekly_songs_played' in df_engineered.columns:
        df_engineered['song_diversity_ratio'] = df_engineered.apply(
            lambda row: row['weekly_unique_songs'] / row['weekly_songs_played'] if row['weekly_songs_played'] > 0 else 0,
            axis=1
        )
    
    # Calculate social activity score
    social_cols = ['num_platform_friends', 'num_playlists_created', 'num_shared_playlists']
    if all(col in df_engineered.columns for col in social_cols):
        max_friends = df_engineered['num_platform_friends'].max() or 1
        max_playlists = df_engineered['num_playlists_created'].max() or 1
        max_shared = df_engineered['num_shared_playlists'].max() or 1
        
        df_engineered['social_activity_score'] = (
            (df_engineered['num_platform_friends'] / max_friends) +
            (df_engineered['num_playlists_created'] / max_playlists) +
            (df_engineered['num_shared_playlists'] / max_shared)
        ) / 3
    
    # Calculate pause frequency normalized by tenure
    if 'num_subscription_pauses' in df_engineered.columns and 'tenure_days' in df_engineered.columns:
        df_engineered['pause_frequency'] = df_engineered.apply(
            lambda row: row['num_subscription_pauses'] / (row['tenure_days'] / 30) if row['tenure_days'] > 0 else 0,
            axis=1
        )
    
    # Calculate notification engagement rate
    if 'notifications_clicked' in df_engineered.columns:
        max_notifications = df_engineered['notifications_clicked'].max() or 1
        df_engineered['notification_engagement_rate'] = df_engineered['notifications_clicked'] / max_notifications
    
    # Calculate combined activity recency score
    activity_cols = ['notification_engagement_rate', 'session_frequency', 'social_activity_score']
    if all(col in df_engineered.columns for col in activity_cols):
        df_engineered['activity_recency_score'] = (
            (df_engineered['notification_engagement_rate'] * 0.4) +
            (df_engineered['session_frequency'] * 0.3) +
            (df_engineered['social_activity_score'] * 0.3)
        )
    
    # Calculate engagement diversity
    if 'engagement_score' in df_engineered.columns and 'song_diversity_ratio' in df_engineered.columns:
        df_engineered['engagement_diversity'] = df_engineered['engagement_score'] * df_engineered['song_diversity_ratio']
    
    # Calculate overall churn risk score
    churn_cols = ['pause_frequency', 'engagement_score', 'customer_service_inquiries_encoded', 'activity_recency_score']
    if all(col in df_engineered.columns for col in churn_cols):
        df_engineered['churn_risk_score'] = (
            df_engineered['pause_frequency'] * 0.3 +
            (1 - df_engineered['engagement_score'] / (df_engineered['engagement_score'].max() or 1)) * 0.2 +
            df_engineered['customer_service_inquiries_encoded'] * 0.2 +
            (1 - df_engineered['activity_recency_score']) * 0.3
        )
    
    return df_engineered