import pandas as pd
import pickle
import os
import requests
import time

def run_inference(model_path='../Membangun_model/model.pkl', data_path='../Membangun_model/dataset_processed.csv'):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Generating future dataframe for inference...")
    future = model.make_future_dataframe(periods=30)
    
    print("Running inference...")
    forecast = model.predict(future)
    
    print("Inference results (tail):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    forecast.to_csv('inference_results.csv', index=False)
    print("Saved inference results to inference_results.csv")

def simulate_traffic():
    print("Simulating traffic to /predict endpoint to generate Prometheus metrics...")
    for i in range(20):
        try:
            requests.post('http://localhost:8000/predict')
            print(f"Request {i+1} sent to /predict")
        except Exception as e:
            print(f"Failed to send request: {e}")
        time.sleep(1.5)

if __name__ == '__main__':
    # run_inference()  # Uncomment when model.pkl is ready
    simulate_traffic()  # Simulate traffic for Grafana/Prometheus screenshots
