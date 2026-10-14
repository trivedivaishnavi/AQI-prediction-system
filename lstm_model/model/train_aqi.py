import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

DATA_DIR = "lstm_model/data"
MODEL_DIR = "lstm_model/model"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LENGTH = 24
PREDICTION_HORIZON = 24

def load_data(city_name):
    df = pd.read_csv(f"{DATA_DIR}/{city_name}.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    df = df.sort_values('timestamp')
    df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
    df = df.dropna(subset=['aqi'])
    return df

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X), np.array(y)

def train_model(city_name):
    print(f"\nTraining model for {city_name}...")
    
    try:
        df = load_data(city_name)
        print(f"Loaded {len(df)} records")
        
        if len(df) < 100:
            print(f"Insufficient data. Need at least 100 records, got {len(df)}")
            return
        
        data = df['aqi'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        X, y = create_sequences(data_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)
        print(f"Created {len(X)} sequences")
        
        if len(X) < 10:
            print(f"Insufficient sequences. Need at least 10, got {len(X)}")
            return
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(PREDICTION_HORIZON)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("Training...")
        history = model.fit(X_train, y_train, epochs=200, batch_size=32, 
                  validation_data=(X_test, y_test), verbose=1)
        
        model.save(f"{MODEL_DIR}/{city_name}_model.h5")
        np.save(f"{MODEL_DIR}/{city_name}_scaler.npy", 
                {'min': scaler.data_min_[0], 'max': scaler.data_max_[0]})
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test MAE: {mae:.2f}")
        print(f"Model saved to {MODEL_DIR}/{city_name}_model.h5")
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    cities = ['delhi', 'mumbai', 'bengaluru']   
    for city in cities:
        train_model(city)   
    print("Training complete!")

if __name__ == "__main__":
    main()