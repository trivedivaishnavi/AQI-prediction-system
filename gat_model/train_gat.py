import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from config import *
from models.gat import GATNet
from data.sample_aqi import SpatiotemporalAQIGenerator

def create_dataset(num_samples=1000):
    """Generate multiple graph snapshots"""
    generator = SpatiotemporalAQIGenerator(num_nodes=NUM_NODES)
    dataset = []   
    for _ in range(num_samples):
        data = generator.generate_next_timestep()
        dataset.append(data)   
    return dataset

def normalize_dataset(dataset):
    """Calculate normalization stats across entire dataset"""
    all_x = torch.cat([data.x for data in dataset], dim=0)
    all_y = torch.cat([data.y for data in dataset], dim=0)   
    x_mean = all_x.mean(dim=0)
    x_std = all_x.std(dim=0) + 1e-6
    y_mean = all_y.mean()
    y_std = all_y.std() + 1e-6

    for data in dataset:
        data.x = (data.x - x_mean) / x_std
        data.y = (data.y - y_mean) / y_std   
    return dataset, x_mean, x_std, y_mean, y_std

def train():
    print("Generating dataset...")
    full_dataset = create_dataset(num_samples=1000)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))   
    train_dataset = full_dataset[:train_size]
    val_dataset = full_dataset[train_size:train_size+val_size]
    test_dataset = full_dataset[train_size+val_size:]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_dataset, x_mean, x_std, y_mean, y_std = normalize_dataset(train_dataset)
    for data in val_dataset + test_dataset:
        data.x = (data.x - x_mean) / x_std
        data.y = (data.y - y_mean) / y_std
        
    torch.save({
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std
    }, "norm_stats.pt")

    model = GATNet(
        in_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        heads=HEADS
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []       
        for data in train_dataset:
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data in val_dataset:
                pred = model(data.x, data.edge_index)
                loss = F.mse_loss(pred, data.y)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "gat_weights.pth")
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
    model.load_state_dict(torch.load("gat_weights.pth"))
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for data in test_dataset:
            pred = model(data.x, data.edge_index)
            loss = F.mse_loss(pred, data.y)
            test_losses.append(loss.item())
    
    test_loss = np.mean(test_losses)
    print(f"\nTest Loss: {test_loss:.6f}")
    
    return model

if __name__ == "__main__":
    trained_model = train()
    print("\nGAT model trained and saved as gat_weights.pth")