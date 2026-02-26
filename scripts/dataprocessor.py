import pandas as pd
import os

# Setup paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')

def process_glucose_dataset(filename):
    # Ensure processed folder exists
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # Load data
    input_file = os.path.join(RAW_PATH, filename)
    if not os.path.exists(input_file):
        print(f"Error: {filename} not found in {RAW_PATH}")
        return

    df = pd.read_csv(input_file)

    # 1. Standardize Timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    # 2. Add 'Menstrual Phase' Logic (Example: Day 1-14 Follicular, 15-28 Luteal)
    # If your data has a 'CycleDay' column, we can map it
    if 'CycleDay' in df.columns:
        df['Phase'] = df['CycleDay'].apply(lambda x: 'Follicular' if x <= 14 else 'Luteal')
    
    # 3. Handle 'Lag Time' (Crucial for Criterion One)
    # We create a column showing what the BGL was 30 mins ago
    # This helps the ML model see the 'Trend'
    df['BG_Trend'] = df['BG_Value'].diff() 
    
    # 4. Filter Noise
    # Remove sensor errors (e.g., readings of 0 or gaps)
    df = df.dropna(subset=['BG_Value'])
    df = df[df['BG_Value'] > 2.0] # Remove unrealistic lows (sensor errors)

    # 5. Export for C++ Consumption
    output_name = f"processed_{filename}"
    df.to_csv(os.path.join(PROCESSED_PATH, output_name), index=False)
    print(f"Success! Processed data saved to: data/processed/{output_name}")

if __name__ == "__main__":
    # Change 'my_data.csv' to whatever your file is named
    process_glucose_dataset('my_data.csv')