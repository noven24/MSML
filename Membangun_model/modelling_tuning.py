import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt

def tune_model(data_path='namadataset_preprocessing/dataset_processed.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run dataset_preprocessing.py first.")
        return

    print("Loading data for tuning...")
    df = pd.read_csv(data_path)
    

    mlflow.set_experiment("Latihan Credit Scoring")
    
    # Mengaktifkan autologging untuk model Prophet (tidak disupport di versi ini)
    # mlflow.autolog()

    # Define hyperparameter grid
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.1],
        'seasonality_prior_scale': [0.1, 1.0]
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []

    print(f"Starting Grid Search for {len(all_params)} parameter combinations...")
    
    # Mulai logging MLflow
    with mlflow.start_run(run_name="Hyperparameter Tuning") as parent_run:
        for params in all_params:
            with mlflow.start_run(nested=True):
                m = Prophet(**params).fit(df)
                
                # Cross-validation (Dipercepat & Fix Deadlock Windows)
                df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days')
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = df_p['rmse'].values[0]
                rmses.append(rmse)
                
                # Autologging merekam metrik
                mlflow.log_metric("rmse", rmse)

        # Mencari parameter terbaik
        best_idx = np.argmin(rmses)
        best_params = all_params[best_idx]
        best_rmse = rmses[best_idx]
        
        print(f"Best RMSE: {best_rmse}")
        print(f"Best Parameters: {best_params}")
        
        # Log Best Parameters pada run parent
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("best_rmse", best_rmse)
        
        # Train ulang dengan parameter terbaik agar modelnya tersimpan sebagai artefak utama
        best_model = Prophet(**best_params).fit(df)

        # ==== Advanced Criteria: Log at least 2 manual artifacts ====
        # Artifact 1: Plot forecast
        print("Generating forecast plot artifact...")
        future = best_model.make_future_dataframe(periods=30)
        forecast = best_model.predict(future)
        fig = best_model.plot(forecast)
        fig.savefig("forecast_plot.png")
        mlflow.log_artifact("forecast_plot.png")
        plt.close(fig)
        
        # Artifact 2: Requirements / Config
        import json
        with open("model_config.json", "w") as f:
            json.dump(best_params, f)
        mlflow.log_artifact("model_config.json")
        
        # We also have requirements.txt in the same folder, let's log it too
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

    return best_params

if __name__ == '__main__':
    tune_model()
