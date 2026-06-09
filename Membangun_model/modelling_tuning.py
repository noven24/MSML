import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
import os
import mlflow
import dagshub
import matplotlib.pyplot as plt

def tune_model(data_path='dataset_processed.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run dataset_preprocessing.py first.")
        return

    print("Loading data for tuning...")
    df = pd.read_csv(data_path)
    
    # ==== Kriteria 2 (Skilled): Localhost Tracking ====
    # mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # ==== Kriteria 2 (Advanced): DagsHub Tracking ====
    # Pastikan Anda memiliki akun DagsHub dan mengganti "noven24/MSML" dengan repository Anda
    # Jika DagsHub meminta kredensial, Anda akan diarahkan untuk login di browser
    dagshub.init(repo_owner='noven24', repo_name='MSML', mlflow=True)
    
    mlflow.set_experiment("MSML_Prophet_Experiment")
    
    # Mengaktifkan autologging untuk model Prophet
    mlflow.prophet.autolog()

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
                
                # Cross-validation
                df_cv = cross_validation(m, initial='180 days', period='30 days', horizon='30 days', parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = df_p['rmse'].values[0]
                rmses.append(rmse)
                
                # Autologging sudah merekam metrik, tetapi kita bisa tambahkan manual
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
        
        # Train ulang dengan parameter terbaik
        best_model = Prophet(**best_params).fit(df)
        
        # ==== Kriteria 2 (Advanced): Minimal dua artefak tambahan ====
        # Artefak 1: File dataset yang digunakan
        mlflow.log_artifact(data_path, artifact_path="data")
        
        # Artefak 2: Plot komponen model Prophet
        fig = best_model.plot_components(best_model.predict(df))
        fig.savefig("prophet_components.png")
        mlflow.log_artifact("prophet_components.png", artifact_path="plots")
        plt.close(fig)

    return best_params

if __name__ == '__main__':
    tune_model()
