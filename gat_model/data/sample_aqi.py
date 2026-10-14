import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from datetime import datetime, timedelta

class SpatiotemporalAQIGenerator:
    def __init__(self, num_nodes=8, start_time=None):
        self.num_nodes = num_nodes
        np.random.seed(42)
        self.coords = np.random.uniform(0, 1000, (num_nodes, 2))
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.linalg.norm(self.coords[i] - self.coords[j]) < 400:
                    edges.append([i, j])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.base_aqi = np.random.uniform(40, 100, num_nodes)
        self.current_pm25 = self.base_aqi.copy()
        self.current_time = start_time or datetime.now()
        self.history = []
        
    def generate_next_timestep(self):
        """Generate next hour with temporal and spatial correlation"""
        hour = self.current_time.hour

        daily_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            daily_factor = 1.3
        elif 2 <= hour <= 5:
            daily_factor = 0.7
            
        new_pm25 = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            temporal_change = np.random.normal(0, 5)
            self_value = self.current_pm25[i] + temporal_change
            neighbors = self.edge_index[1][self.edge_index[0] == i].numpy()
            if len(neighbors) > 0:
                neighbor_avg = np.mean(self.current_pm25[neighbors])
                new_pm25[i] = 0.7 * self_value + 0.3 * neighbor_avg
            else:
                new_pm25[i] = self_value
            new_pm25[i] *= daily_factor
            new_pm25[i] = np.clip(new_pm25[i], 20, 200)
        
        if np.random.random() < 0.05: 
            affected_node = np.random.randint(0, self.num_nodes)
            new_pm25[affected_node] += np.random.uniform(20, 50)

        self.current_pm25 = new_pm25
        pm10 = new_pm25 + np.random.uniform(5, 40, self.num_nodes)
        mq135 = new_pm25 * 2 + np.random.uniform(50, 150, self.num_nodes)
        temp = np.random.uniform(20, 40, self.num_nodes)
        humidity = np.random.uniform(30, 80, self.num_nodes)
        x = np.stack([new_pm25, pm10, mq135, temp, humidity], axis=1)
        x = torch.tensor(x, dtype=torch.float)
        
        y = 0.6 * new_pm25 + 0.3 * pm10 + 0.1 * (mq135 / 3)
        y = torch.tensor(y, dtype=torch.float)
        
        data = Data(x=x, edge_index=self.edge_index, y=y)

        self.history.append({
            'timestamp': self.current_time,
            'data': data,
            'pm25': new_pm25.copy()
        })

        self.current_time += timedelta(hours=1)       
        return data
    
    def generate_dataset(self, num_hours=720):
        """Generate dataset for specified hours"""
        for _ in range(num_hours):
            self.generate_next_timestep()
        return self.history
    
    def save_to_csv(self, filename='aqi_log.csv'):
        """Save to CSV format matching your schema"""
        records = []
        
        for entry in self.history:
            timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            data = entry['data']
            
            for node_id in range(self.num_nodes):
                records.append({
                    'timestamp': timestamp,
                    'node_id': node_id,
                    'place_name': f'Area_{node_id}',
                    'aqi': float(data.y[node_id])
                })      
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        print(f"Saved {len(records)} records to {filename}")

generator = SpatiotemporalAQIGenerator(num_nodes=8)
generator.generate_dataset(num_hours=720)  # 30 days
generator.save_to_csv('aqi_log.csv')