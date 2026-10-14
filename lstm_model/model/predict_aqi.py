import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

DATA_DIR = "lstm_model/data"
MODEL_DIR = "lstm_model/model"

SEQUENCE_LENGTH = 24

def load_data(city_name):
    df = pd.read_csv(f"{DATA_DIR}/{city_name}.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    df = df.sort_values('timestamp')
    df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
    df = df.dropna(subset=['aqi'])
    return df

def predict_aqi(city_name, hours=24):
    print(f"\nPredicting AQI for {city_name}...")
    
    df = load_data(city_name)
    print(f"Using last {SEQUENCE_LENGTH} hours of data")
    
    model = tf.keras.models.load_model(f"{MODEL_DIR}/{city_name}_model.h5", compile=False)
    scaler_params = np.load(f"{MODEL_DIR}/{city_name}_scaler.npy", allow_pickle=True).item()
    
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array([scaler_params['min']])
    scaler.data_max_ = np.array([scaler_params['max']])
    scaler.scale_ = np.array([1.0 / (scaler_params['max'] - scaler_params['min'])])
    scaler.min_ = np.array([0.0])
    scaler.data_range_ = np.array([scaler_params['max'] - scaler_params['min']])
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = 1

    
    data = df['aqi'].values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    
    last_seq = data_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    pred_scaled = model.predict(last_seq, verbose=0)
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    last_time = df['timestamp'].iloc[-1]
    future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(hours)]
    
    result = pd.DataFrame({
        'timestamp': future_times[:len(predictions)],
        'predicted_aqi': predictions[:hours]
    })
    
    print(f"\nPredictions for next {hours} hours:")
    print(result.to_string(index=False))
    
    result.to_csv(f"{MODEL_DIR}/{city_name}_predictions.csv", index=False)
    print(f"\nSaved to {MODEL_DIR}/{city_name}_predictions.csv")
    
    return result

def main():
    cities = ['delhi', 'mumbai', 'bengaluru']
    
    print("="*60)
    print("LSTM AQI Prediction")
    print("="*60)
    
    hours = int(input("\nEnter hours to predict (12 or 24): "))
    
    for city in cities:
        try:
            predict_aqi(city, hours)
        except Exception as e:
            print(f"Error predicting {city}: {e}")
    
    print("\n" + "="*60)
    print("Predictions complete!")
    print("="*60)

if __name__ == "__main__":
    main()