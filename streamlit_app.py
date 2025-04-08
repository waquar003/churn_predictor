import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

st.title("🎵 Customer Churn Prediction App")

st.write("Enter the customer’s data below to predict churn.")

# Input Fields
age = st.slider("Age", 10, 100)
num_subscription_pauses = st.number_input("Number of Subscription Pauses", 0, 50)
weekly_hours = st.slider("Weekly Hours", 0.0, 100.0)
average_session_length = st.slider("Average Session Length (seconds)", 0.0, 500.0)
song_skip_rate = st.slider("Song Skip Rate", 0.0, 1.0)
weekly_songs_played = st.number_input("Weekly Songs Played", 0, 1000)
weekly_unique_songs = st.number_input("Weekly Unique Songs", 0, 1000)
num_favorite_artists = st.number_input("Number of Favorite Artists", 0, 100)
num_platform_friends = st.number_input("Number of Platform Friends", 0, 500)
num_playlists_created = st.number_input("Playlists Created", 0, 100)
num_shared_playlists = st.number_input("Shared Playlists", 0, 100)
notifications_clicked = st.number_input("Notifications Clicked", 0, 100)

churned = st.selectbox("Previously Churned?", [0, 1])
location_encoded = st.selectbox("Location Encoded", [0, 1, 2])  # Adjust if more classes

# Subscription (One-hot)
subscription_Family = st.selectbox("Subscription: Family Plan", [0, 1])
subscription_Free = st.selectbox("Subscription: Free Plan", [0, 1])
subscription_Premium = st.selectbox("Subscription: Premium Plan", [0, 1])
subscription_Student = st.selectbox("Subscription: Student Plan", [0, 1])

payment_plan_encoded = st.selectbox("Payment Plan Encoded", [0, 1, 2])  # Adjust based on your dataset

# Payment methods (One-hot)
payment_Apple = st.selectbox("Apple Pay", [0, 1])
payment_Credit = st.selectbox("Credit Card", [0, 1])
payment_Debit = st.selectbox("Debit Card", [0, 1])
payment_Paypal = st.selectbox("Paypal", [0, 1])

customer_service_inquiries_encoded = st.number_input("Customer Service Inquiries (Encoded)", 0, 10)
tenure_days = st.number_input("Tenure (days)", 0, 5000)

signup_recency = st.slider("Signup Recency", 0.0, 1.0)
engagement_score = st.slider("Engagement Score", 0.0, 100.0)
session_frequency = st.slider("Session Frequency", 0.0, 1.0)
song_diversity_ratio = st.slider("Song Diversity Ratio", 0.0, 10.0)
social_activity_score = st.slider("Social Activity Score", 0.0, 1.0)
pause_frequency = st.slider("Pause Frequency", 0.0, 1.0)
notification_engagement_rate = st.slider("Notification Engagement Rate", 0.0, 1.0)
activity_recency_score = st.slider("Activity Recency Score", 0.0, 1.0)
engagement_diversity = st.slider("Engagement Diversity", 0.0, 20.0)

# Combine all input features
features = np.array([[
    age, num_subscription_pauses, weekly_hours, average_session_length,
    song_skip_rate, weekly_songs_played, weekly_unique_songs,
    num_favorite_artists, num_platform_friends, num_playlists_created,
    num_shared_playlists, notifications_clicked, churned,
    location_encoded, subscription_Family, subscription_Free,
    subscription_Premium, subscription_Student, payment_plan_encoded,
    payment_Apple, payment_Credit, payment_Debit, payment_Paypal,
    customer_service_inquiries_encoded, tenure_days, signup_recency,
    engagement_score, session_frequency, song_diversity_ratio,
    social_activity_score, pause_frequency, notification_engagement_rate,
    activity_recency_score, engagement_diversity
]])

# Predict churn
if st.button("Predict Churn"):
    prediction = model.predict(features)
    result = "⚠️ Likely to Churn" if prediction[0] == 1 else "✅ Likely to Stay"
    st.success(f"Prediction: {result}")