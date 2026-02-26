import pandas as pd
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')

def process_medtronic_data(filename):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    input_file = os.path.join(RAW_PATH, filename)
    
    # 1. Load data skipping the first 6 rows of metadata
    df = pd.read_csv(input_file, skiprows=6, low_memory=False)

    # 2. Create a proper Timestamp
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('Timestamp')

    # 3. Clean up the Columns
    # Medtronic uses NaN for 'no event'. We fill these strategically.
    
    # Sensor Glucose: Use interpolation to fill the 5-minute gaps
    df['SG'] = pd.to_numeric(df['Sensor Glucose (mmol/L)'], errors='coerce')
    df['SG'] = df['SG'].interpolate(method='time')

    # Insulin & Carbs: Fill empty with 0 (since no event means 0 delivered)
    df['Bolus'] = pd.to_numeric(df['Bolus Volume Delivered (U)'], errors='coerce').fillna(0)
    df['Carbs'] = pd.to_numeric(df['BWZ Carb Input (grams)'], errors='coerce').fillna(0)
    
    # Basal: Forward fill (it stays the same until the next change)
    df['Basal'] = pd.to_numeric(df['Basal Rate (U/h)'], errors='coerce').ffill()

    # 4. Add your SAT Features (Menstrual Cycle)
    # You can manually set the start date of your cycle here to auto-calculate the phase
    cycle_start_date = pd.to_datetime('2026-02-13') # Change this to your actual start date
    df['Days_Into_Cycle'] = (df['Timestamp'] - cycle_start_date).dt.days % 28 + 1
    df['Phase'] = df['Days_Into_Cycle'].apply(lambda x: 'Follicular' if x <= 14 else 'Luteal')

    # 5. Select only relevant columns for the ML Base
    final_columns = ['Timestamp', 'SG', 'Bolus', 'Carbs', 'Basal', 'Days_Into_Cycle', 'Phase']
    df_final = df[final_columns].dropna(subset=['SG']) # Only keep rows where we have glucose data

    output_name = "cleaned_medtronic_data.csv"
    df_final.to_csv(os.path.join(PROCESSED_PATH, output_name), index=False)
    print(f"Success! Created a clean dataset with {len(df_final)} entries.")

if __name__ == "__main__":
    process_medtronic_data('Milana Kumykova 26-02-2026.csv')