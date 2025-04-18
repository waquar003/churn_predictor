import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from PIL import Image
import sys
import io
import base64

# Import local modules
sys.path.append('..')
from src.data.preprocessing import preprocess_data
from src.data.features import engineer_features
from src.models.predict import predict_churn, get_churn_factors, load_model
from src.visualization.plots import (
    plot_churn_distribution, plot_risk_distribution, plot_feature_importance,
    plot_churn_by_category, plot_churn_probability_distribution, plot_customer_profile,
)

# Set page configuration
st.set_page_config(
    page_title="Music Streaming Customer Churn Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    /* Base styling */
    body {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
    }
    .main {
        background-color: #f1f5f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 600;
    }
    h1 {
        font-size: 32px;
        margin-bottom: 20px;
    }
    h2 {
        font-size: 24px;
        margin-bottom: 16px;
        margin-top: 0px !important;
        margin-bottom: 8px !important;
    }
    h3 {
        font-size: 20px;
        margin-bottom: 12px;
        margin-top: 0px !important;
        margin-bottom: 8px !important;
    }
    p {
        line-height: 1.6;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        gap: 1px;
        padding: 10px 15px;
        font-weight: 500;
        color: #64748b;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #1e3a8a;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: white !important;
    }
    
    /* Cards */
    .card {
        border-radius: 8px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    .card-header {
        padding-bottom: 10px;
        margin-bottom: 15px;
        border-bottom: 1px solid #e2e8f0;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
        color: #1e293b !important;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk categories */
    .high-risk {
        color: #dc2626;
        background-color: #fee2e2;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .moderate-risk {
        color: #d97706;
        background-color: #fef3c7;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .low-risk {
        color: #16a34a;
        background-color: #dcfce7;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Filter section */
    .filter-section {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #1e40af;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 600;
        padding: 10px !important;
        text-align: left;
    }
    .dataframe td {
        padding: 8px 10px !important;
        border-bottom: 1px solid #e2e8f0;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }
    .dataframe tr:hover {
        background-color: #e2e8f0;
    }
    
    /* Widgets */
    .stSlider [data-baseweb=slider] {
        margin-top: 10px;
    }
    .stSlider [data-baseweb=slider] [data-testid=stThumbValue] {
        background-color: #1e3a8a;
        color: white;
    }
    .stMultiSelect [data-baseweb=select] {
        background-color: white;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
    .stMultiSelect [data-baseweb=tag] {
        background-color: #1e3a8a;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    .sidebar-content {
        padding: 15px;
    }
    [data-testid=stSidebar] [data-testid=stHeader] {
        background-color: #1e3a8a;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Welcome screen */
    .welcome-title {
        font-size: 28px;
        margin-bottom: 20px;
        color: #1e3a8a;
    }
    .welcome-card {
        padding: 30px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    .feature-list {
        margin: 20px 0;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .feature-icon {
        color: #1e3a8a;
        margin-right: 10px;
    }
    
    /* Customer details section */
    .customer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
    }
    .customer-risk-indicator {
        font-size: 16px;
        font-weight: 600;
        padding: 5px 10px;
        border-radius: 6px;
    }
    .customer-detail-section {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 15px;
        margin-bottom: 15px;
    }
    .customer-detail-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #334155;
    }
    
    /* Recommendations */
    .recommendation-item {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #1e3a8a;
    }
    .recommendation-title {
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 5px;
    }
    .recommendation-action {
        background-color: #e2e8f0;
        padding: 5px 10px;
        border-radius: 4px;
        color: #334155;
        display: inline-block;
        margin-top: 5px;
        font-size: 14px;
    }
    
    /* Risk factors */
    .risk-factor-item {
        padding: 10px;
        border-radius: 6px;
        background-color: #f8fafc;
        margin-bottom: 10px;
        border-left: 3px solid #1e3a8a;
    }
    .risk-factor-high {
        border-left-color: #dc2626;
    }
    .risk-factor-moderate {
        border-left-color: #d97706;
    }
    .risk-factor-low {
        border-left-color: #16a34a;
    }
    .risk-factor-title {
        font-weight: 600;
        margin-bottom: 5px;
    }
    .risk-factor-stats {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
    }
    
    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th {
        background-color: #f1f5f9;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        color: #334155;
        border-bottom: 2px solid #cbd5e1;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    /* Chart container */
    .chart-container {
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    /* Text colors for better visibility */
    .text-dark {
        color: #1E293B;
    }
    .text-light {
        color: #64748B;
    }
    .text-accent {
        color: #1e3a8a;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a8a;
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #1e3a8a;
        color: white;
    }
    
    /* Header with background */
    .header-section {
        background-color: #1e3a8a;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .header-section h1 {
        color: white;
        margin: 0;
    }
    
    /* Metrics section */
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9;
        color: #0c4a6e;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        color: #7c2d12;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'active_customer_id' not in st.session_state:
    st.session_state.active_customer_id = None


MODEL_PATH = 'models/churn_model_v1.pkl'

pd.set_option("styler.render.max_elements", 50000)

def main():
    st.title("🎵 Music Streaming Customer Churn Predictor")
    
    # Create sidebar
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload customer data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the data
                df = pd.read_csv(uploaded_file)
                
                # Store in session state
                st.session_state.data = df
                
                # Display data info
                with st.expander("Data Overview"):
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    st.text(info_str)
                    
                # Process button
                st.markdown("### Process Data")
                process_col1, process_col2 = st.columns(2)
                with process_col1:
                    if st.button("Preprocess Data"):
                        with st.spinner("Preprocessing data..."):
                            # Preprocess data
                            preprocessed_df = preprocess_data(df)
                            # Engineer features
                            engineered_df = engineer_features(preprocessed_df)
                            # Store in session state
                            st.session_state.processed_data = engineered_df
                            st.success("Data preprocessed successfully!")
                
                if st.session_state.processed_data is not None:
                    if st.button("Make Predictions"):
                        with st.spinner("Making predictions..."):
                            # Make predictions
                            st.session_state.model_data = load_model(MODEL_PATH)
                            predictions = predict_churn(
                                st.session_state.processed_data, 
                                model_data=st.session_state.model_data,
                                model_path=MODEL_PATH
                            )
                            # Store in session state
                            st.session_state.predictions = predictions
                            st.success("Predictions made successfully!")
                
                # Customer search
                if st.session_state.predictions is not None:
                    st.markdown("### Customer Search")
                    customer_id = st.number_input(
                        "Enter Customer ID", 
                        min_value=int(st.session_state.predictions['customer_id'].min()),
                        max_value=int(st.session_state.predictions['customer_id'].max())
                    )
                    
                    if st.button("Search Customer"):
                        # Set active customer ID
                        st.session_state.active_customer_id = customer_id
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This application predicts customer churn for a music streaming service "
            "using machine learning. Upload customer data, analyze patterns, and identify "
            "customers at risk of churning."
        )

    # Main content area
    if st.session_state.data is None:
        # Display welcome screen
        st.markdown(
            """
            <div class="card" style="padding: 20px; background-color: #f1f5f9; border-radius: 8px;">
                <h2 style="font-size: 32px; font-weight: 600; color: #1e293b;">Welcome to the Music Streaming Customer Churn Predictor</h2>
                <p style="font-size: 18px; color: #4b5563;">This application helps you predict and analyze customer churn for a music streaming service.</p>
                <h3 style="font-size: 24px; color: #1e293b;">Features:</h3>
                <ul style="font-size: 18px; color: #4b5563;">
                    <li>Upload and analyze customer data</li>
                    <li>Visualize data distributions and patterns</li>
                    <li>Train machine learning models to predict churn</li>
                    <li>Identify customers at risk of churning</li>
                    <li>Get insights into churn factors</li>
                </ul>
                <p style="font-size: 18px; color: #4b5563;">To get started, upload a customer data CSV file using the sidebar.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Sample data structure for reference
        st.markdown("### Expected Data Structure")
        sample_data = pd.DataFrame({
            'customer_id': [1001, 1002, 1003],
            'age': [34.0, 27.0, 45.0],
            'location': ['New York', 'Los Angeles', 'Chicago'],
            'subscription_type': ['Premium', 'Free', 'Family'],
            'payment_plan': ['Monthly', 'Monthly', 'Yearly'],
            'num_subscription_pauses': [1, 0, 2],
            'payment_method': ['Credit Card', 'Paypal', 'Apple Pay'],
            'customer_service_inquiries': ['Low', 'Medium', 'High'],
            'signup_date': [365, 180, 90],  # Days since reference date
            'weekly_hours': [10.5, 5.2, 8.7],
            'average_session_length': [25.3, 18.7, 30.1],
            'song_skip_rate': [0.15, 0.32, 0.08],
            'weekly_songs_played': [120, 75, 180],
            'weekly_unique_songs': [80, 45, 130],
            'num_favorite_artists': [15, 8, 22],
            'num_platform_friends': [5, 12, 3],
            'num_playlists_created': [8, 3, 15],
            'num_shared_playlists': [2, 0, 7],
            'notifications_clicked': [25, 10, 35],
        })
        st.dataframe(sample_data)
    
    else:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview", 
            "🔎 Churn Analysis", 
            "🧮 Predictions", 
            "👤 Customer Details"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Data Overview")
            
            # Display basic dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Customers", len(st.session_state.data))
            with col2:
                missing_values = st.session_state.data.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(10))
            
            # Missing values visualization
            st.subheader("Missing Values Analysis")
            missing_data = st.session_state.data.isnull().sum().reset_index()
            missing_data.columns = ['Feature', 'Missing Count']
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            if not missing_data.empty:
                missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(st.session_state.data) * 100).round(2)
                fig = px.bar(
                    missing_data,
                    x='Feature',
                    y='Missing Count',
                    text='Missing Percentage',
                    color='Missing Percentage',
                    color_continuous_scale='Reds',
                    title='Missing Values by Feature'
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found in the dataset.")
            
        # Tab 2: Churn Analysis
        with tab2:
            st.header("Churn Analysis")
            
            if st.session_state.predictions is not None:
                predictions = st.session_state.predictions
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    churn_rate = predictions['churned'].mean() * 100 if 'churned' in predictions.columns else predictions['predicted_churn'].mean() * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Churn Rate</div>
                            <div class="metric-value">{churn_rate:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    high_risk = (predictions['churn_risk_category'] == 'High').mean() * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">High Risk Customers</div>
                            <div class="metric-value high-risk">{high_risk:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col3:
                    moderate_risk = (predictions['churn_risk_category'] == 'Moderate').mean() * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Moderate Risk Customers</div>
                            <div class="metric-value moderate-risk">{moderate_risk:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col4:
                    low_risk = (predictions['churn_risk_category'] == 'Low').mean() * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Low Risk Customers</div>
                            <div class="metric-value low-risk">{low_risk:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Churn distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    churn_dist_fig = plot_churn_distribution(predictions)
                    st.plotly_chart(churn_dist_fig, use_container_width=True)
                
                with col2:
                    risk_dist_fig = plot_risk_distribution(predictions)
                    st.plotly_chart(risk_dist_fig, use_container_width=True)
                
                # Churn probability distribution
                prob_fig = plot_churn_probability_distribution(predictions)
                st.plotly_chart(prob_fig, use_container_width=True)
                
                # Feature importance
                if st.session_state.model_data is not None:
                    fi_fig = plot_feature_importance(st.session_state.model_data)
                    st.plotly_chart(fi_fig, use_container_width=True)
                
                # Categorical feature analysis
                st.subheader("Churn Analysis by Categories")
                
                # Determine categorical features
                categorical_cols = [
                    'subscription_type', 'payment_plan', 'payment_method', 
                    'customer_service_inquiries', 'location'
                ]
                
                available_cat_cols = [col for col in categorical_cols if col in predictions.columns]
                
                if available_cat_cols:
                    selected_cat = st.selectbox(
                        "Select categorical feature:",
                        options=available_cat_cols
                    )
                    
                    if selected_cat:
                        cat_fig = plot_churn_by_category(predictions, selected_cat)
                        st.plotly_chart(cat_fig, use_container_width=True)
            else:
                st.info("Please process data and generate predictions to view churn analysis.")
        
        # Tab 3: Predictions
        with tab3:
            st.header("Customer Churn Predictions")
            
            if st.session_state.predictions is not None:
                predictions = st.session_state.predictions
                
                # Filters
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.subheader("Filter Customers")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_filter = st.multiselect(
                        "Risk Category:",
                        options=['High', 'Moderate', 'Low'],
                        default=['High', 'Moderate', 'Low']
                    )
                
                with col2:
                    if 'subscription_type' in predictions.columns:
                        subscription_options = sorted(predictions['subscription_type'].unique().tolist())
                        subscription_filter = st.multiselect(
                            "Subscription Type:",
                            options=subscription_options,
                            default=subscription_options
                        )
                    else:
                        subscription_filter = None
                
                with col3:
                    if 'payment_method' in predictions.columns:
                        payment_options = sorted(predictions['payment_method'].unique().tolist())
                        payment_filter = st.multiselect(
                            "Payment Method:",
                            options=payment_options,
                            default=payment_options
                        )
                    else:
                        payment_filter = None
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Make sure to convert to float to avoid any issues with sliders
                    min_prob = float(predictions['churn_probability'].min())
                    max_prob = float(predictions['churn_probability'].max())
                    prob_range = st.slider(
                        "Churn Probability Range:",
                        min_value=min_prob,
                        max_value=max_prob,
                        value=(min_prob, max_prob),
                        step=0.01
                    )
                
                sort_options = {
                    'Customer ID': 'customer_id',
                    'Churn Probability (High to Low)': 'churn_probability',
                    'Churn Probability (Low to High)': '-churn_probability'
                }
                
                with col2:
                    sort_by = st.selectbox(
                        "Sort By:",
                        options=list(sort_options.keys()),
                        index=1
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply filters
                filtered_df = predictions.copy()
                
                # Risk category filter
                if risk_filter:
                    filtered_df = filtered_df[filtered_df['churn_risk_category'].isin(risk_filter)]
                
                # Subscription filter
                if subscription_filter is not None and 'subscription_type' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['subscription_type'].isin(subscription_filter)]
                
                # Payment method filter
                if payment_filter is not None and 'payment_method' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['payment_method'].isin(payment_filter)]
                
                # Probability range filter
                filtered_df = filtered_df[
                    (filtered_df['churn_probability'] >= prob_range[0]) & 
                    (filtered_df['churn_probability'] <= prob_range[1])
                ]
                
                # Sort
                sort_col = sort_options[sort_by]
                if sort_col.startswith('-'):
                    filtered_df = filtered_df.sort_values(sort_col[1:], ascending=True)
                else:
                    filtered_df = filtered_df.sort_values(sort_col, ascending=False)
                
                # Display filtered data
                st.subheader(f"Filtered Customers ({len(filtered_df)} results)")
                
                # Check if filtered_df is empty
                if filtered_df.empty:
                    st.warning("No customers match the selected filters. Please adjust your criteria.")
                else:
                    # Select columns to display
                    display_cols = ['customer_id', 'churn_probability', 'churn_risk_category']
                    
                    # Add original categorical columns if available
                    cat_cols = ['subscription_type', 'payment_plan', 'payment_method', 'location']
                    display_cols.extend([col for col in cat_cols if col in filtered_df.columns])
                    
                    # Add some engagement metrics
                    engagement_cols = ['engagement_score', 'social_activity_score', 'weekly_hours']
                    display_cols.extend([col for col in engagement_cols if col in filtered_df.columns])
                    
                    # Make sure we only include columns that actually exist in the dataframe
                    display_cols = [col for col in display_cols if col in filtered_df.columns]
                    
                    # Format the churn probability
                    formatted_df = filtered_df[display_cols].copy()
                    if 'churn_probability' in formatted_df.columns:
                        formatted_df['churn_probability'] = formatted_df['churn_probability'].map('{:.1%}'.format)
                    
                    # Apply color to risk category
                    def highlight_risk(val):
                        if val == 'High':
                            return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold'
                        elif val == 'Moderate':
                            return 'background-color: #fff9c4; color: #f57f17; font-weight: bold'
                        elif val == 'Low':
                            return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold'
                        return ''
                    
                    # Display the styled dataframe
                    if 'churn_risk_category' in formatted_df.columns:
                        st.dataframe(
                            formatted_df.style.apply(
                                lambda x: [highlight_risk(val) if col == 'churn_risk_category' else '' 
                                        for col, val in zip(x.index, x.values)],
                                axis=1
                            ),
                            height=400,
                            use_container_width=True
                        )
                    else:
                        st.dataframe(formatted_df, height=400, use_container_width=True)
                    
                    # Download CSV button
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Filtered Results",
                        data=csv,
                        file_name="filtered_churn_predictions.csv",
                        mime="text/csv",
                        key='download-csv'
                    )
            else:
                st.info("Please process data and generate predictions to view customer predictions.")
        
        # Tab 4: Customer Details
        with tab4:
            st.markdown("<h2 class='text-accent'>Customer Details</h2>", unsafe_allow_html=True)
            
            if st.session_state.active_customer_id is not None and st.session_state.predictions is not None:
                customer_id = st.session_state.active_customer_id
                predictions = st.session_state.predictions
                
                # Get customer data
                customer_data = predictions[predictions['customer_id'] == customer_id]
                
                if not customer_data.empty:
                    customer = customer_data.iloc[0]
                    risk_category = customer['churn_risk_category']
                    risk_class = 'high-risk' if risk_category == 'High' else ('moderate-risk' if risk_category == 'Moderate' else 'low-risk')
                    
                    # Customer header with risk indicator
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="customer-header" style="display: flex; align-items: center; justify-content: space-between; background-color: #f3f4f6; padding: 15px; border-radius: 8px;">
                                <h3 style="font-size: 24px; font-weight: 600; color: #1e293b;">Customer #{customer['customer_id']}</h3>
                                <span class="customer-risk-indicator {risk_class}" style="font-size: 14px; padding: 5px 10px; border-radius: 12px; background-color: #ef4444; color: white;">
                                    {risk_category} Risk
                                </span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    
                    # Customer profile
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Key metrics
                        st.markdown("<div class='customer-detail-section'>", unsafe_allow_html=True)
                        st.markdown("<div class='customer-detail-title'>Churn Risk</div>", unsafe_allow_html=True)
                        
                        churn_prob = customer['churn_probability'] * 100
                        churn_color = "#dc2626" if churn_prob > 70 else ("#d97706" if churn_prob > 30 else "#16a34a")
                        
                        # Progress bar for churn probability
                        st.markdown(
                            f"""
                            <div style="margin-bottom: 20px;">
                                <div style="font-weight: bold; margin-bottom: 5px; color: {churn_color};">
                                    Churn Probability: {churn_prob:.1f}%
                                </div>
                                <div style="background-color: #f1f5f9; border-radius: 10px; height: 10px; position: relative;">
                                    <div style="position: absolute; background-color: {churn_color}; width: {churn_prob}%; height: 10px; border-radius: 10px;"></div>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Basic info
                        st.markdown("<div class='customer-detail-section'>", unsafe_allow_html=True)
                        st.markdown("<div class='customer-detail-title'>Basic Information</div>", unsafe_allow_html=True)
                        
                        # Display basic information in a clean format
                        if 'age' in customer:
                            st.markdown(f"<p><strong>Age:</strong> {customer['age']}</p>", unsafe_allow_html=True)
                        if 'location' in customer:
                            st.markdown(f"<p><strong>Location:</strong> {customer['location']}</p>", unsafe_allow_html=True)
                        if 'subscription_type' in customer:
                            st.markdown(f"<p><strong>Subscription:</strong> {customer['subscription_type']}</p>", unsafe_allow_html=True)
                        if 'payment_method' in customer:
                            st.markdown(f"<p><strong>Payment Method:</strong> {customer['payment_method']}</p>", unsafe_allow_html=True)
                        if 'signup_date' in customer:
                            st.markdown(f"<p><strong>Days Since Signup:</strong> {customer['signup_date']}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Usage metrics
                        st.markdown("<div class='customer-detail-section'>", unsafe_allow_html=True)
                        st.markdown("<div class='customer-detail-title'>Usage Metrics</div>", unsafe_allow_html=True)
                        
                        usage_metrics = [
                            ('weekly_hours', 'Weekly Hours'),
                            ('average_session_length', 'Avg. Session (min)'),
                            ('weekly_songs_played', 'Weekly Songs'),
                            ('song_skip_rate', 'Skip Rate'),
                            ('num_playlists_created', 'Playlists Created'),
                            ('num_favorite_artists', 'Favorite Artists')
                        ]
                        
                        for col, label in usage_metrics:
                            if col in customer:
                                value = customer[col]
                                if col == 'song_skip_rate':
                                    value = f"{value * 100:.1f}%"
                                elif col == 'average_session_length':
                                    value = f"{value:.1f}"
                                st.markdown(f"<p><strong>{label}:</strong> {value}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Customer profile visualization
                        st.markdown("<div class='customer-detail-title'>Customer Profile</div>", unsafe_allow_html=True)
                        profile_fig = plot_customer_profile(customer)
                        profile_fig.update_layout(
                            height=400,
                            margin=dict(t=0, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#334155')
                        )
                        st.plotly_chart(profile_fig, use_container_width=True)
                        
                        # Risk factors
                        st.markdown("<div class='customer-detail-title'>Risk Factors</div>", unsafe_allow_html=True)
                        
                        # Get risk factors
                        risk_factors = get_churn_factors(df=st.session_state.predictions, customer_id=customer['customer_id'], model_data=st.session_state.model_data)
                    
                    # Recommendations section
                    st.markdown("<div class='customer-detail-title'>Recommendations</div>", unsafe_allow_html=True)
                    
                    # Generate personalized recommendations based on risk factors
                    if risk_category == 'High':
                        recommendations = [
                            {
                                "title": "Immediate Outreach",
                                "description": "Contact the customer directly with a personalized retention offer.",
                                "action": "Offer 3 months at 50% discount"
                            },
                            {
                                "title": "Feature Education",
                                "description": "Provide guidance on features the customer is underutilizing.",
                                "action": "Send feature highlight email"
                            },
                            {
                                "title": "Feedback Request",
                                "description": "Request specific feedback on pain points or issues.",
                                "action": "Send short satisfaction survey"
                            }
                        ]
                    elif risk_category == 'Moderate':
                        recommendations = [
                            {
                                "title": "Proactive Engagement",
                                "description": "Increase engagement with personalized content recommendations.",
                                "action": "Send curated playlist"
                            },
                            {
                                "title": "Mild Incentive",
                                "description": "Offer a small incentive to improve engagement.",
                                "action": "Offer 1 month free premium feature"
                            }
                        ]
                    else:
                        recommendations = [
                            {
                                "title": "Regular Monitoring",
                                "description": "Continue monitoring usage patterns for any changes.",
                                "action": "Set quarterly review"
                            },
                            {
                                "title": "Loyalty Recognition",
                                "description": "Acknowledge customer loyalty to reinforce relationship.",
                                "action": "Send appreciation message"
                            }
                        ]
                    
                    for rec in recommendations:
                        st.markdown(
                            f"""
                            <div class="recommendation-item" style="color: #334155;">
                                <div class="recommendation-title" style="font-weight: bold; color: #1e293b;">{rec['title']}</div>
                                <p style="color: #4b5563;">{rec['description']}</p>
                                <span class="recommendation-action" style="color: #1e40af; font-weight: bold;">{rec['action']}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="warning-box">Customer not found. Please check the ID and try again.</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="info-box">Please select a customer to view details.</div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()