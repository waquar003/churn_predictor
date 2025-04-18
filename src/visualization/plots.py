import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def plot_churn_distribution(df):
    """
    Plot churn distribution
    
    Args:
        df: DataFrame with churn data
        
    Returns:
        fig: Plotly figure
    """
    if 'churned' not in df.columns and 'predicted_churn' not in df.columns:
        raise ValueError("No churn column found in DataFrame")
    
    # Determine which column to use
    churn_col = 'churned' if 'churned' in df.columns else 'predicted_churn'
    
    # Calculate churn counts
    churn_counts = df[churn_col].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    churn_counts['Churn'] = churn_counts['Churn'].map({0: 'Not Churned', 1: 'Churned'})
    
    # Calculate percentages
    total = churn_counts['Count'].sum()
    churn_counts['Percentage'] = (churn_counts['Count'] / total * 100).round(1)
    
    # Create plot
    fig = px.pie(
        churn_counts, 
        values='Count',
        names='Churn',
        title='Customer Churn Distribution',
        color='Churn',
        color_discrete_map={'Churned': '#FF5757', 'Not Churned': '#4CAF50'},
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}'
    )
    
    return fig

def plot_risk_distribution(df):
    """
    Plot churn risk distribution
    
    Args:
        df: DataFrame with churn risk data
        
    Returns:
        fig: Plotly figure
    """
    if 'churn_risk_category' not in df.columns:
        raise ValueError("No churn risk category column found in DataFrame")
    
    # Calculate risk counts
    risk_counts = df['churn_risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    
    # Sort risk categories in meaningful order
    risk_order = {'Low': 0, 'Moderate': 1, 'High': 2}
    risk_counts['Sort'] = risk_counts['Risk Category'].map(risk_order)
    risk_counts = risk_counts.sort_values('Sort').drop('Sort', axis=1)
    
    # Calculate percentages
    total = risk_counts['Count'].sum()
    risk_counts['Percentage'] = (risk_counts['Count'] / total * 100).round(1)
    
    # Define colors for risk categories
    color_map = {'Low': '#4CAF50', 'Moderate': '#FFC107', 'High': '#FF5757'}
    
    # Create plot
    fig = px.bar(
        risk_counts,
        x='Risk Category',
        y='Count',
        title='Churn Risk Distribution',
        color='Risk Category',
        color_discrete_map=color_map,
        text='Percentage'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title='Churn Risk Category',
        yaxis_title='Count',
        showlegend=False
    )
    
    return fig

def plot_feature_importance(model_data):
    """
    Plot feature importance
    
    Args:
        model_data: Model data dictionary with feature importance
        
    Returns:
        fig: Plotly figure
    """
    # Extract feature importance
    features = model_data['feature_importance']['features']
    importance = model_data['feature_importance']['importance']
    
    # Create DataFrame for plotting
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Sort by importance
    fi_df = fi_df.sort_values('Importance', ascending=False)
    
    # Take top 15 features for cleaner visualization
    fi_df = fi_df.head(15)
    
    # Create plot
    fig = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        title='Top 15 Feature Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_churn_by_category(df, category_col):
    """
    Plot churn rate by a categorical feature
    
    Args:
        df: DataFrame with churn data
        category_col: Categorical column name
        
    Returns:
        fig: Plotly figure
    """
    if category_col not in df.columns:
        raise ValueError(f"Column {category_col} not found in DataFrame")
    
    # Determine which column to use for churn
    churn_col = 'churned' if 'churned' in df.columns else 'predicted_churn'
    if churn_col not in df.columns:
        raise ValueError("No churn column found in DataFrame")
    
    # Calculate churn rate by category
    churn_by_cat = df.groupby(category_col)[churn_col].mean().reset_index()
    churn_by_cat.columns = [category_col, 'Churn Rate']
    
    # Add count for each category
    cat_counts = df[category_col].value_counts().reset_index()
    cat_counts.columns = [category_col, 'Count']
    churn_by_cat = churn_by_cat.merge(cat_counts, on=category_col)
    
    # Sort by churn rate
    churn_by_cat = churn_by_cat.sort_values('Churn Rate', ascending=False)
    
    # Create plot
    fig = px.bar(
        churn_by_cat,
        x=category_col,
        y='Churn Rate',
        title=f'Churn Rate by {category_col}',
        color='Churn Rate',
        color_continuous_scale='RdYlGn_r',
        text='Count'
    )
    
    fig.update_traces(
        texttemplate='Count: %{text}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title='Churn Rate',
        yaxis_tickformat='.1%'
    )
    
    return fig

def plot_churn_probability_distribution(df):
    """
    Plot distribution of churn probabilities
    
    Args:
        df: DataFrame with churn probability
        
    Returns:
        fig: Plotly figure
    """
    if 'churn_probability' not in df.columns:
        raise ValueError("No churn probability column found in DataFrame")
    
    # Create histogram
    fig = px.histogram(
        df,
        x='churn_probability',
        nbins=30,
        title='Distribution of Churn Probabilities',
        color_discrete_sequence=['#6495ED']
    )
    
    # Add vertical lines for risk categories
    fig.add_shape(
        type='line',
        x0=0.3, x1=0.3,
        y0=0, y1=1,
        yref='paper',
        line=dict(color='green', width=2, dash='dash')
    )
    
    fig.add_shape(
        type='line',
        x0=0.7, x1=0.7,
        y0=0, y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash')
    )
    
    # Add annotations for risk categories
    fig.add_annotation(
        x=0.15, y=0.95,
        yref='paper',
        text='Low Risk',
        showarrow=False,
        bgcolor='rgba(76, 175, 80, 0.3)',
        bordercolor='green',
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.5, y=0.95,
        yref='paper',
        text='Moderate Risk',
        showarrow=False,
        bgcolor='rgba(255, 193, 7, 0.3)',
        bordercolor='orange',
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.85, y=0.95,
        yref='paper',
        text='High Risk',
        showarrow=False,
        bgcolor='rgba(255, 87, 87, 0.3)',
        bordercolor='red',
        borderwidth=1
    )
    
    fig.update_layout(
        xaxis_title='Churn Probability',
        yaxis_title='Count'
    )
    
    return fig

def plot_customer_profile(customer_data):
    """
    Create a radar chart for customer profile visualization
    
    Args:
        customer_data: DataFrame row with customer data
        
    Returns:
        fig: Plotly figure
    """
    # Select relevant metrics for the profile
    metrics = [
        'engagement_score', 
        'social_activity_score',
        'song_diversity_ratio',
        'session_frequency',
        'notification_engagement_rate'
    ]
    
    # Make sure all metrics exist
    metrics = [m for m in metrics if m in customer_data.index]
    
    if len(metrics) == 0:
        raise ValueError("No profile metrics found in customer data")
    
    # Extract values
    values = [customer_data[m] for m in metrics]
    
    # Normalize values between 0 and 1
    max_values = {
        'engagement_score': 30,
        'social_activity_score': 1,
        'song_diversity_ratio': 1,
        'session_frequency': 20,
        'notification_engagement_rate': 1
    }
    
    normalized_values = [min(values[i] / max_values.get(metrics[i], 1), 1) for i in range(len(metrics))]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Customer Profile',
        line_color='royalblue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Customer Engagement Profile',
        showlegend=False
    )
    
    return fig

