import pandas as pd
import pickle
import os

def run_inference(model_path='../Membangun_model/model.pkl', data_path='../Membangun_model/dataset_processed.csv'):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Normally we would take new data, but for this template we just forecast into the future
    print("Generating future dataframe for inference...")
    future = model.make_future_dataframe(periods=30)
    
    print("Running inference...")
    forecast = model.predict(future)
    
    print("Inference results (tail):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    forecast.to_csv('inference_results.csv', index=False)
    print("Saved inference results to inference_results.csv")

if __name__ == '__main__':
    run_inference()
