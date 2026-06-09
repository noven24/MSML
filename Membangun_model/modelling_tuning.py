import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np

def tune_model(data_path='dataset_processed.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run dataset_preprocessing.py first.")
        return

    print("Loading data for tuning...")
    df = pd.read_csv(data_path)
    
    # Define hyperparameter grid
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []

    print(f"Starting Grid Search for {len(all_params)} parameter combinations...")
    
    # Normally we would do full CV, but to keep it fast for this template, 
    # we'll just demonstrate the structure of a tuning loop.
    for params in all_params:
        m = Prophet(**params).fit(df)  # Fit model with given params
        
        # Cross-validation
        # Note: initial, period, horizon should be adjusted based on dataset span
        df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    best_params = all_params[np.argmin(rmses)]
    print(f"Best RMSE: {min(rmses)}")
    print(f"Best Parameters: {best_params}")
    
    return best_params

if __name__ == '__main__':
    # This might take a while depending on the data size
    import os
    tune_model()
