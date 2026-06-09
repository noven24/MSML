import pandas as pd
from prophet import Prophet
import pickle
import os

def train_model(data_path='dataset_processed.csv', model_path='model.pkl'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run dataset_preprocessing.py first.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Training Prophet model...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model training complete.")

if __name__ == '__main__':
    train_model()
