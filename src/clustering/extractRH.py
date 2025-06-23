import os
import pandas as pd
from tqdm import tqdm
import json
import shutil

def read_geojson_to_df(path):
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['features'])
    return df, data

def get_save_path(df, i):
    """Get the save path for weather station data"""
    return os.path.join(SAVE_PATH1, df['properties'][i]['save_path'])

def load_weather_data(df, i):
    """Load weather data for a specific station"""
    save_path = get_save_path(df, i)
    try:
        data = pd.read_csv(save_path, header=None, names=['Date', 'Value'])
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        print(f"Error loading weather data from {save_path}: {e}")
        return None

def find_nearest_point(points, point):
    """Find the nearest weather station to the given coordinates"""
    min_dist = float('inf')
    min_idx = -1
    for i, p in enumerate(points):
        if not isinstance(p, list) or len(p) != 2:
            continue
        dist = (p[0] - point[0])**2 + (p[1] - point[1])**2
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx

def find_nearest_date(df, date):
    """Find the nearest date in weather data and return humidity value"""
    if df is None or df.empty:
        return None
        
    date = pd.to_datetime(date)
    if date in df['Date'].values:
        return df[df['Date'] == date]['Value'].values[0]
    else:
        idx = df['Date'].sub(date).abs().idxmin()
        return df.loc[idx]['Value']

def load_event_data(file_path):
    """Load event CSV data file with proper cleaning"""
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Remove rows with missing coordinates
        data = data.dropna(subset=['Latitude', 'Longitude'])
        
        # Filter for Hong Kong area
        hk_filter = (
            (data['Latitude'] > 20) & 
            (data['Latitude'] < 25) & 
            (data['Longitude'] > 113) & 
            (data['Longitude'] < 115)
        )
        return data[hk_filter].copy()
    except Exception as e:
        print(f"Error loading event data from {file_path}: {e}")
        return None

def st_match(year):
    """Match humidity data to events for a specific year"""
    # Load metadata for weather stations
    try:
        df, _ = read_geojson_to_df(os.path.join(SAVE_PATH1, 'metadata.json'))
        # Extract coordinates from GeoJSON
        points = df['geometry'].apply(lambda x: x['coordinates'] if x and 'coordinates' in x else None)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Load event data
    input_file = os.path.join(SAVE_DIR, f'{year}.csv')
    output_file = os.path.join(SAVE_DIR, f'{year}.csv')
    
    # Make a backup copy of the original file
    if not os.path.exists(output_file):
        shutil.copy2(input_file, output_file)
    
    data = load_event_data(output_file)
    if data is None:
        return

    # Add new column for humidity
    if 'Humidity' not in data.columns:
        data['Humidity'] = None

    # Process each event
    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {year}"):
        point = (row['Latitude'], row['Longitude'])
        date = row['Date']
        
        # Find nearest weather station
        nearest_station_idx = find_nearest_point(points, point)
        if nearest_station_idx == -1:
            continue
            
        # Load weather data for that station
        weather_data = load_weather_data(df, nearest_station_idx)
        if weather_data is None:
            continue
            
        # Find humidity for nearest date
        humidity = find_nearest_date(weather_data, date)
        data.at[idx, 'Humidity'] = humidity

    # Save to new file without overwriting original
    data.to_csv(output_file, index=False)
    print(f"Saved humidity-matched data to {output_file}")

if __name__ == '__main__':
    # Set up paths
    DIR = os.path.dirname(os.path.abspath(__file__))
    WDIR = os.path.join(DIR, '..', '..', 'data2', 'weather') 
    SAVE_PATH1 = os.path.join(WDIR, 'output', 'RH')
    SAVE_DIR = os.path.join(DIR, '..', '..', 'data2', 'files')

    # Process years
    YEARS = [2021, 2022, 2023, 2024]
    
    for year in YEARS:
        st_match(year)