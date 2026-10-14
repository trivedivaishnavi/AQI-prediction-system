import torch
import csv
import os
from datetime import datetime

from config import NUM_NODES, NUM_FEATURES, HIDDEN_DIM, OUT_DIM, HEADS
from models.gat import GATNet
from data.sample_aqi import SpatiotemporalAQIGenerator

NODE_NAMES = [
    "City Center",
    "Industrial Area",
    "Residential Zone",
    "Highway Junction",
    "Commercial Hub",
    "Park Area",
    "Suburban Area",
    "Outskirts"
]

CSV_FILE = "aqi_log.csv"

def init_csv():
    """Create CSV file with header if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "node_id", "place_name", "aqi"])

def load_model():
    """Load trained GAT model"""
    model = GATNet(
        in_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        heads=HEADS
    )
    model.load_state_dict(torch.load("gat_weights.pth", map_location="cpu"))
    model.eval()
    
    norm_stats = torch.load("norm_stats.pt")
    return model, norm_stats

def predict_aqi(model, data, norm_stats):
    """Predict AQI for current graph snapshot"""
    data.x = (data.x - norm_stats["x_mean"]) / norm_stats["x_std"]

    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
    pred_real = predictions * norm_stats["y_std"] + norm_stats["y_mean"]
    return pred_real

def evaluate_current():
    """Evaluate current AQI snapshot"""
    init_csv()

    model, norm_stats = load_model()
    generator = SpatiotemporalAQIGenerator(num_nodes=NUM_NODES)
    data = generator.generate_next_timestep()
    pred_aqi = predict_aqi(model, data, norm_stats)
    print("\n" + "="*60)
    print("CURRENT AQI PREDICTIONS")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i in range(NUM_NODES):
        aqi_value = pred_aqi[i].item()
        print(f"{NODE_NAMES[i]:<20} → AQI: {aqi_value:7.2f}")
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        for i in range(NUM_NODES):
            writer.writerow([
                timestamp,
                i,
                NODE_NAMES[i],
                round(pred_aqi[i].item(), 4)
            ])
    
    print(f"\nLogged to {CSV_FILE}")

def generate_historical_data(hours=720):
    """Generate historical AQI data for LSTM training"""
    print(f"\nGenerating {hours} hours of historical data...")
    init_csv()
    
    model, norm_stats = load_model()
    generator = SpatiotemporalAQIGenerator(num_nodes=NUM_NODES)
    
    for hour in range(hours):
        data = generator.generate_next_timestep()
        pred_aqi = predict_aqi(model, data, norm_stats)
        
        timestamp = generator.current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i in range(NUM_NODES):
                writer.writerow([
                    timestamp,
                    i,
                    NODE_NAMES[i],
                    round(pred_aqi[i].item(), 4)
                ])
        
        if (hour + 1) % 100 == 0:
            print(f"Generated {hour + 1}/{hours} hours")
    
    print(f"\nCompleted! {hours * NUM_NODES} records saved to {CSV_FILE}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 720
        generate_historical_data(hours)
    else:
        evaluate_current()