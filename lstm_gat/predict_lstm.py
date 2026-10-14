import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:   
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.config = None
        self.node_data = {}
        
    def load_models(self):
        try:
            with open(f'{self.model_dir}/config.pkl', 'rb') as f:
                self.config = pickle.load(f)
            
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.keras'):
                    model_key = filename.replace('.keras', '')
                    self.models[model_key] = keras.models.load_model(
                        f'{self.model_dir}/{filename}'
                    )
                    
                    scaler_file = f'{self.model_dir}/{model_key}_scaler.pkl'
                    with open(scaler_file, 'rb') as f:
                        self.scalers[model_key] = pickle.load(f)
            
            print(f"Models loaded successfully\n")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def load_data(self, csv_path='aqi_log.csv'):
        try:
            df = pd.read_csv(csv_path)
            
            for node_id in range(self.config['nodes']):
                node_df = df[df['node_id'] == node_id].sort_values('timestamp')
                if len(node_df) > 0:
                    self.node_data[node_id] = node_df['aqi'].values
            
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def predict(self, node_id, prediction_horizon):
        model_key = f"node_{node_id}_{prediction_horizon}h"
        
        if model_key not in self.models:
            print(f"Error: Model not found for node {node_id}")
            return None
        
        if node_id not in self.node_data:
            print(f"Error: No data for node {node_id}")
            return None
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        sequence_length = self.config['sequence_length']       
        recent_data = self.node_data[node_id][-sequence_length:]
        recent_scaled = scaler.transform(recent_data.reshape(-1, 1))
        X = recent_scaled.reshape(1, sequence_length, 1)       
        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
        
        return prediction

if __name__ == "__main__":
    print("="*60)
    print("LSTM AQI PREDICTION")
    print("="*60)  
    predictor = LSTMPredictor('saved_models')  
    if not predictor.load_models():
        exit()
    
    if not predictor.load_data('aqi_log.csv'):
        exit()
          
    while True:
        print("\n" + "="*60)        
        node_input = input("Enter node ID (0-7) or 'q' to quit: ").strip()
        
        if node_input.lower() == 'q':
            break
        
        try:
            node_id = int(node_input)
            if node_id < 0 or node_id > 7:
                print("Invalid node ID. Use 0-7")
                continue
        except ValueError:
            print("Invalid input")
            continue
        
        horizon_input = input("Enter prediction horizon (12 or 24): ").strip()
        
        try:
            horizon = int(horizon_input)
            if horizon not in [12, 24]:
                print("Invalid horizon. Use 12 or 24")
                continue
        except ValueError:
            print("Invalid input")
            continue        
        prediction = predictor.predict(node_id, horizon)       
        if prediction is not None:
            print(f"\nPredicted AQI for Node {node_id} ({horizon}h): {prediction:.2f}")   
    print("\nThank You")