import requests
import csv
import os
from datetime import datetime, timedelta, timezone
import time

WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "YOUR_WAQI_TOKEN_HERE")
cities = {
    "delhi": "delhi",
    "mumbai": "mumbai",
    "bengaluru": "bengaluru"
}
DATA_DIR = os.path.join("lstm_model", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_current_aqi(city):
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "ok":
            return data["data"]
    return None

def extract_data_point(city_name, aqi_data):
    now = datetime.now(timezone.utc)
    iaqi = aqi_data.get("iaqi", {})
    
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "city": city_name.title(),
        "aqi": aqi_data.get("aqi", "N/A"),
        "pm25": iaqi.get("pm25", {}).get("v", "N/A"),
        "pm10": iaqi.get("pm10", {}).get("v", "N/A"),
        "o3": iaqi.get("o3", {}).get("v", "N/A"),
        "no2": iaqi.get("no2", {}).get("v", "N/A"),
        "so2": iaqi.get("so2", {}).get("v", "N/A"),
        "co": iaqi.get("co", {}).get("v", "N/A"),
        "temperature": iaqi.get("t", {}).get("v", "N/A"),
        "humidity": iaqi.get("h", {}).get("v", "N/A"),
        "pressure": iaqi.get("p", {}).get("v", "N/A"),
        "wind_speed": iaqi.get("w", {}).get("v", "N/A")
    }

def save_csv(city_key, new_point):
    filename = os.path.join(DATA_DIR, f"{city_key}.csv")
    existing_data = []
    
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=60)
        def parse_timestamp(ts):
            try:
                return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            return None

        existing_data = [
            row for row in existing_data 
            if parse_timestamp(row["timestamp"]) and parse_timestamp(row["timestamp"]) > cutoff_date
        ]
        
    if any(row["timestamp"] == new_point["timestamp"] for row in existing_data):
        return len(existing_data)
    
    existing_data.append(new_point)
    existing_data.sort(key=lambda x: x["timestamp"])
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_point.keys())
        writer.writeheader()
        writer.writerows(existing_data)
    
    return len(existing_data)

def collect_once():
    print(f"Collecting data at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for city_key, city_name in cities.items():
        aqi_data = fetch_current_aqi(city_name)
        if aqi_data:
            data_point = extract_data_point(city_name, aqi_data)
            total = save_csv(city_key, data_point)
            print(f"{city_key.title()}: AQI={data_point['aqi']}, Total records={total}")
        else:
            print(f"{city_key.title()}: Failed to fetch")
        time.sleep(1)
    print()

def main():
    if WAQI_TOKEN == "YOUR_WAQI_TOKEN_HERE":
        print(" Get a WAQI token ")
        return
    
    print()
    
    collect_once()    
    print()
    
    choice = input("Keep running and collect every hour? (y/n): ").lower()  
    
    if choice == 'y':
        print("\nCollecting data every hour. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(3600)
                collect_once()
        except KeyboardInterrupt:
            print("\nStopped by user")

if __name__ == "__main__":
    main()