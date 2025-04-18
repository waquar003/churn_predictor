import os
from src.data.preprocessing import load_data, preprocess_data
from src.data.features import engineer_features
from src.models.train import train_and_save_model

def main():
    # Define paths
    raw_data_path = "data/raw/Dataset.csv"
    model_output_path = "models/churn_model_v2.pkl"
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    print("📥 Loading data...")
    df_raw = load_data(raw_data_path)

    print("🧹 Preprocessing data...")
    df_preprocessed = preprocess_data(df_raw)

    print("🛠️ Engineering features...")
    df_engineered = engineer_features(df_preprocessed)

    print("📊 Training model...")
    model_data = train_and_save_model(df_engineered, model_path=model_output_path)

    print("✅ Model training complete and saved to:", model_output_path)
    print("📈 Performance metrics:")
    for metric, value in model_data['metrics'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
