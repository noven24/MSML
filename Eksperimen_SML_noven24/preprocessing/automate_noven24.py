import pandas as pd
import os
import argparse

def preprocess_data(input_path, output_path):
    print(f"Loading raw dataset from {input_path}...")
    try:
        df_raw = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return None

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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Processed dataset shape: {df_prophet.shape}")
    df_prophet.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return df_prophet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing")
    parser.add_argument('--input', type=str, default='namadataset_raw/dataset.csv', help='Path to raw dataset')
    parser.add_argument('--output', type=str, default='preprocessing/namadataset_preprocessing/dataset_processed.csv', help='Path to save processed dataset')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)
