import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')

def process_medtronic_data(filename):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    input_file = os.path.join(RAW_PATH, filename)
    
    if not os.path.exists(input_file):
        print(f"Error: {filename} not found.")
        return

    # 1. Load data
    df = pd.read_csv(input_file, skiprows=6, low_memory=False)

    # 2. Clean Timestamps (The fix for your error)
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['Timestamp']) # Remove the rows causing the crash
    df = df.sort_values('Timestamp')

    # 3. Process Values
    # Convert to numeric, turn errors to NaN, then fill/interpolate
    df['SG'] = pd.to_numeric(df['Sensor Glucose (mmol/L)'], errors='coerce')
    df['SG'] = df['SG'].interpolate(method='linear') 

    df['Bolus'] = pd.to_numeric(df['Bolus Volume Delivered (U)'], errors='coerce').fillna(0)
    df['Carbs'] = pd.to_numeric(df['BWZ Carb Input (grams)'], errors='coerce').fillna(0)
    df['Basal'] = pd.to_numeric(df['Basal Rate (U/h)'], errors='coerce').ffill().fillna(0)

    # 4. Cycle Phase Logic
    cycle_start_date = pd.to_datetime('2026-02-13') 
    df['Days_Into_Cycle'] = (df['Timestamp'] - cycle_start_date).dt.days % 28 + 1
    df['Phase'] = df['Days_Into_Cycle'].apply(lambda x: 'Follicular' if x <= 14 else 'Luteal')

    # 5. Export
    final_columns = ['Timestamp', 'SG', 'Bolus', 'Carbs', 'Basal', 'Days_Into_Cycle', 'Phase']
    df_final = df[final_columns].dropna(subset=['SG'])

    output_name = "cleaned_medtronic_data.csv"
    df_final.to_csv(os.path.join(PROCESSED_PATH, output_name), index=False)
    print(f"Success! Cleaned {len(df_final)} rows of glucose data.")

if __name__ == '__main__':
    process_medtronic_data('Milana Kumykova 26-02-2026.csv')