import pandas as pd
import json

def extract_geo_data(csv_path, output_path):
    df = pd.read_csv(csv_path)
    # Ensure numeric types
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Sort by index (assuming chronological order)
    df = df.reset_index(drop=True)
    
    # Simple risk zone definition matching notebook Logic
    high_risk_zone_bounds = {'lat_min': 30, 'lat_max': 40, 'long_min': -100, 'long_max': -90}
    df['is_high_risk'] = ((df['Latitude'] >= high_risk_zone_bounds['lat_min']) &
                          (df['Latitude'] <= high_risk_zone_bounds['lat_max']) &
                          (df['Longitude'] >= high_risk_zone_bounds['long_min']) &
                          (df['Longitude'] <= high_risk_zone_bounds['long_max'])).astype(int)
    
    # Group into flight segments (10 records per flight)
    segment_size = 10
    total_rows = len(df)
    
    def get_region(lat, lon):
        if 24 <= lat <= 49 and -125 <= lon <= -66: return "USA"
        if 35 <= lat <= 70 and -10 <= lon <= 40: return "Europe"
        if -35 <= lat <= 5 and -75 <= lon <= -35: return "South America"
        if 0 <= lat <= 60 and 60 <= lon <= 150: return "Asia"
        return "International"

    flights = []
    # Loop to cover everything, including remainder
    for i in range(0, total_rows, segment_size):
        segment = df.iloc[i : i + segment_size]
        path = segment[['Latitude', 'Longitude']].values.tolist()
        is_high_risk = int(segment['is_high_risk'].max())
        
        # Determine region from first point
        region = get_region(path[0][0], path[0][1])
        
        flights.append({
            'id': (i // segment_size) + 1,
            'path': path,
            'is_high_risk': is_high_risk,
            'region': region
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flights, f)
    
    print(f"Extracted {len(flights)} flight trajectories (100% coverage, {total_rows} rows) with regional tags to {output_path}")

if __name__ == "__main__":
    extract_geo_data("data prep/meta_learningdata/final_acas_data.csv", "dashboard/geo_data.json")
