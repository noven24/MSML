import pandas as pd
import os

def preprocess_data(input_path='../../dataset.csv', output_path='dataset_processed.csv'):
    print(f"Loading dataset from {input_path}...")
    df_raw = pd.read_csv(input_path)
    
    print("Formatting dates...")
    df_raw['ds'] = pd.to_datetime(df_raw['Time Date'].astype(str).str.zfill(8), format='%d%m%Y', errors='coerce')
    
    print("Filtering data for Product=2667437 and Store='QLD_CW_ST0203'...")
    df_prophet = df_raw.loc[(df_raw['Product'] == 2667437) & (df_raw['Store'] == 'QLD_CW_ST0203')].copy()
    
    print("Renaming Value to y...")
    df_prophet = df_prophet.rename(columns={'Value': 'y'})
    df_prophet = df_prophet[['ds', 'y']].sort_values('ds')
    
    print("Dropping NA and duplicates...")
    df_prophet = df_prophet.dropna(subset=['ds'])
    df_prophet = df_prophet.drop_duplicates()
    
    print(f"Processed dataset shape: {df_prophet.shape}")
    df_prophet.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return df_prophet

if __name__ == '__main__':
    preprocess_data()
