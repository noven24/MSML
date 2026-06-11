import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def tune_model(data_path='namadataset_preprocessing/insurance_preprocessed.csv'):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Pastikan Anda menjalankan kode ini dari folder yang benar.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Fitur dan Target untuk dataset insurance
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set nama eksperimen
    mlflow.set_experiment("Tuning Insurance Model")
    
    # Kriteria Wajib: Menggunakan autolog dari MLflow
    print("Mengaktifkan MLflow autolog...")
    mlflow.autolog()

    # Define hyperparameter grid untuk RandomForest
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }

    print("Mulai Hyperparameter Tuning dengan GridSearchCV...")
    with mlflow.start_run(run_name="GridSearch_Tuning"):
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Fit GridSearchCV (MLflow autolog akan merekam semua metric, param, dan best_model otomatis)
        grid_search.fit(X_train, y_train)
        
        # Prediksi menggunakan model terbaik
        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        
        print(f"Hyperparameter Tuning Selesai!")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Model RMSE pada data test: {rmse}")
        
    print("Selesai! Model beserta artifact (MLmodel, conda.yaml, model.pkl, dll) telah dicatat ke MLflow.")

if __name__ == '__main__':
    tune_model()
