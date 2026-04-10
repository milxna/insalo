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

    #load data
    df = pd.read_csv(input_file, skiprows=6, low_memory=False)

    #clean timestamps
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['Timestamp']) 
    df = df.sort_values('Timestamp')

    #process values 
    #convert to numeric, handle errors, and interpolate missing glucose values
    df['SG'] = pd.to_numeric(df['Sensor Glucose (mmol/L)'], errors='coerce')
    df['SG'] = df['SG'].interpolate(method='linear') 

    df['Bolus'] = pd.to_numeric(df['Bolus Volume Delivered (U)'], errors='coerce').fillna(0)
    df['Carbs'] = pd.to_numeric(df['BWZ Carb Input (grams)'], errors='coerce').fillna(0)
    df['Basal'] = pd.to_numeric(df['Basal Rate (U/h)'], errors='coerce').ffill().fillna(0)

    #insert cycle phase stuffz
    cycle_start_date = pd.to_datetime('2026-03-28') 
    df['Days_Into_Cycle'] = (df['Timestamp'] - cycle_start_date).dt.days % 28 + 1
    df['Phase'] = df['Days_Into_Cycle'].apply(lambda x: 'Follicular' if x <= 14 else 'Luteal')

    #push to cleaned csv 
    final_columns = ['Timestamp', 'SG', 'Bolus', 'Carbs', 'Basal', 'Days_Into_Cycle', 'Phase']
    df_final = df[final_columns].dropna(subset=['SG'])

    output_name = "cleaned_medtronic_data.csv"
    df_final.to_csv(os.path.join(PROCESSED_PATH, output_name), index=False)
    print(f"Success! Cleaned {len(df_final)} rows of glucose data.")

if __name__ == '__main__':
    process_medtronic_data('Milana Kumykova 28-03-2026.csv')