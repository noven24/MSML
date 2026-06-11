import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(data_path='namadataset_preprocessing/insurance_preprocessed.csv'):
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
    mlflow.set_experiment("Latihan Credit Scoring Kriteria 2")
    
    # Kriteria Wajib: Menggunakan autolog dari MLflow
    print("Mengaktifkan MLflow autolog...")
    mlflow.autolog()

    print("Melatih model Scikit-Learn (RandomForest)...")
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        print(f"Model berhasil dilatih dengan RMSE: {rmse}")
        
    print("Selesai! Model dan metrik telah dicatat ke MLflow.")

if __name__ == '__main__':
    train_model()
