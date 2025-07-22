import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from config import config
from config import SimpleNN

load_path = config['data_path']
data = np.load(load_path)
print(data.shape)
N = config['N']
samples = data.shape[0]
in_dim = N  # X_data.shape[1]
out_dim = 2 * N
# dim = 1

inputs = torch.tensor(data[:, :N], dtype=torch.float32)
outputs = torch.tensor(data[:, N:], dtype=torch.float32)
dataset = TensorDataset(inputs, outputs)

# Define split ratio (e.g., 80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Randomly split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = int(samples / 100)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_norm = SimpleNN(input_size=in_dim, output_size=out_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model_norm = model_norm.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model_local_single.parameters(), lr=1e-6, weight_decay=0)
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
optimizer = torch.optim.Adam(model_norm.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Lists to store loss values
train_losses = []
val_losses = []
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #0.124
epoch_losses = []
# Training loop
num_epochs = config['epochs']  # num_epochs
for epoch in range(num_epochs):
    # model_local.train()  # Set the model to training mode
    epoch_loss = 0.0
    for batch_inputs, batch_outputs in train_loader:
        batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
        predictions = model_norm(batch_inputs)  # Forward pass
        loss = criterion(predictions, batch_outputs)  # Compute loss

        optimizer.zero_grad()  # Zero the parameter gradients

        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        # scheduler.step(loss.item())
        epoch_loss += loss.item() / len(batch_inputs)
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    if (epoch + 1) % 25 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs},Train Loss: {avg_epoch_loss:.8f}')

    model_norm.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model_norm(inputs)
            loss = criterion(predictions, targets)
            val_loss += loss.item() / len(inputs)

    val_losses.append(val_loss / len(val_loader))  # S
    if (epoch + 1) % 25 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs},Val Loss: {val_loss / len(val_loader):.8f}')

model_save_path = config['model_path']
torch.save(model_norm.state_dict(), model_save_path)

model_norm = model_norm.to('cpu')

offline_inputs = inputs
offline_outputs = outputs

plot_save_path = config['train_plot_path']


# Plotting the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='s', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path, "training_loss"))
#plt.show()

# Plotting the loss over epochs (log scale)
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.semilogy(range(1, num_epochs + 1), val_losses, marker='s', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss Over Epochs (Log Scale)')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig(os.path.join(plot_save_path, "training_loss_logscale"))

#plt.show()