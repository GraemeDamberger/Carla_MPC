import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from config import SimpleNN
from config import config
from Shared.logging_utils import save_plot, save_model
import numpy as np

def train(trial_num,log_dir):
    load_path = config['data_path']
    data = np.load(load_path)
    print(data.shape)
    N = config['N']
    N_des = config['N_des']
    in_dim = N+2*N_des  # X_data.shape[1]
    out_dim = 2 * N

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    inputs = torch.tensor(data[:, :in_dim], dtype=torch.float32)
    outputs = torch.tensor(data[:, in_dim:], dtype=torch.float32)
    inputs = inputs.to(device)
    outputs = outputs.to(device)

    # -----------------------------
    # Create train/val split
    dataset = TensorDataset(inputs, outputs)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # -----------------------------
    # Define model
    model = SimpleNN(input_size=in_dim, output_size=out_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = model.to(device)
    #model = SimpleNN().to(device)

    # -----------------------------
    # Training setup
    criterion = nn.MSELoss()
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = config['epochs']
    train_losses = []
    val_losses = []

    # -----------------------------
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ---- Validation step ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred_val = model(x_val)
                val_loss = criterion(y_pred_val, y_val)
                total_val_loss += val_loss.item() * x_val.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        if (epoch + 1) % 25 == 0:
            print(f"[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    model_save_path = config['model_path']
    torch.save(model.state_dict(), model_save_path)

    # Plot (linear scale)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    save_plot(log_dir, fig, f"training_loss_trial_{trial_num}")
    #plt.show()

    # Plot (log scale)
    fig = plt.figure(figsize=(10, 5))
    plt.semilogy(train_losses, label="Train Loss")
    plt.semilogy(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training vs Validation Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    save_plot(log_dir,fig,f"training_loss_logscale_{trial_num}")

    save_model(log_dir,model,f"model_trial_{trial_num}")