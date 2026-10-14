import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

class LSTMTrainer:  
    def __init__(self, sequence_length=24, nodes=8):
        self.sequence_length = sequence_length
        self.nodes = nodes
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def load_data(self, csv_path='aqi_log.csv'):
        try:
            df = pd.read_csv(csv_path)
            print(f"Data loaded: {len(df)} records")
            
            self.node_data = {}
            for node_id in range(self.nodes):
                node_df = df[df['node_id'] == node_id].sort_values('timestamp')
                if len(node_df) > 0:
                    self.node_data[node_id] = node_df['aqi'].values
                    print(f"Node {node_id}: {len(node_df)} records")
            
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def prepare_sequences(self, node_data, prediction_horizon=12):
        X, y = [], []
        for i in range(len(node_data) - self.sequence_length - prediction_horizon):
            X.append(node_data[i:i + self.sequence_length])
            y.append(np.mean(node_data[i + self.sequence_length:
                                       i + self.sequence_length + prediction_horizon]))
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
            layers.Dropout(0.2),
            layers.LSTM(16),
            layers.Dropout(0.2),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        return model
    
    def calculate_accuracy(self, y_true, y_pred, threshold=10):
        """Calculate percentage of predictions within threshold"""
        errors = np.abs(y_true - y_pred)
        within_threshold = np.sum(errors <= threshold)
        accuracy = (within_threshold / len(y_true)) * 100
        return accuracy
    
    def train_node_model(self, node_id, prediction_horizon=12, epochs=100):
        if node_id not in self.node_data:
            print(f"Skipping node {node_id} - no data")
            return None
        
        node_values = self.node_data[node_id].reshape(-1, 1)
        scaler = MinMaxScaler()
        node_data_scaled = scaler.fit_transform(node_values)
        X, y = self.prepare_sequences(node_data_scaled.flatten(), prediction_horizon)
        
        if len(X) < 50:
            print(f"Skipping node {node_id} - insufficient data ({len(X)} samples)")
            return None
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = self.build_model((self.sequence_length, 1))
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        print(f"Training node {node_id} ({prediction_horizon}h)...")       
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        accuracy = self.calculate_accuracy(y_test_actual, y_pred_actual, threshold=10)
        
        print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | Accuracy: {accuracy:.1f}%")
        
        model_key = f"node_{node_id}_{prediction_horizon}h"
        self.models[model_key] = model
        self.scalers[model_key] = scaler

        self.metrics[model_key] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy),
            'test_samples': int(len(y_test))
        }
        
        return model
    
    def train_all(self, epochs=100):
        print("\nTraining all nodes for 12h and 24h predictions...\n")
        
        for node_id in range(self.nodes):
            for horizon in [12, 24]:
                self.train_node_model(node_id, prediction_horizon=horizon, epochs=epochs)
        
        self.print_summary()
    
    def print_summary(self):
        """Print overall accuracy summary"""
        if not self.metrics:
            return
        
        print("\n" + "="*70)
        print("OVERALL MODEL ACCURACY")
        print("="*70)
        metrics_12h = {k: v for k, v in self.metrics.items() if '12h' in k}
        metrics_24h = {k: v for k, v in self.metrics.items() if '24h' in k}
        
        for horizon_name, horizon_metrics in [('12-Hour', metrics_12h), ('24-Hour', metrics_24h)]:
            if not horizon_metrics:
                continue
            
            mae_values = [m['mae'] for m in horizon_metrics.values()]
            accuracy_values = [m['accuracy'] for m in horizon_metrics.values()]
            
            print(f"\n{horizon_name} Predictions:")
            print(f"  Average MAE: ±{np.mean(mae_values):.2f} AQI units")
            print(f"  Average Accuracy (±10 AQI): {np.mean(accuracy_values):.1f}%")
        
        print("\n" + "="*70)
    
    def save_models(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        
        for model_key, model in self.models.items():
            model.save(f'{model_dir}/{model_key}.keras')
            with open(f'{model_dir}/{model_key}_scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers[model_key], f)
        
        config = {
            'sequence_length': self.sequence_length,
            'nodes': self.nodes
        }
        with open(f'{model_dir}/config.pkl', 'wb') as f:
            pickle.dump(config, f)
        with open(f'{model_dir}/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nModels and metrics saved to {model_dir}/")

if __name__ == "__main__":
    print("="*60)
    print("LSTM AQI MODEL TRAINING")
    print("="*60)
    
    trainer = LSTMTrainer(sequence_length=24, nodes=8)
    
    if trainer.load_data('aqi_log.csv'):
        trainer.train_all(epochs=100)
        trainer.save_models('saved_models')
        print("\nTraining complete!")
    else:
        print("Failed to load data")