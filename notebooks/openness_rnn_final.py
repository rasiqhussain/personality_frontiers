# In[1]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load embeddings and labels
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)

with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)

with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)

with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

with open('train_seq_len', 'rb') as handle:
    train_seq_len = pickle.load(handle)

with open('test_seq_len', 'rb') as handle:
    test_seq_len = pickle.load(handle)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(longitudinal_train, dtype=torch.float32)
X_test_tensor = torch.tensor(longitudinal_test, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.float32)

train_seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_train], dtype=torch.int64)
test_seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_test], dtype=torch.int64)

# Combine train and test for cross-validation
X = torch.cat((X_train_tensor, X_test_tensor), dim=0)
y = torch.cat((y_train_tensor, y_test_tensor), dim=0)
seq_len = torch.cat((train_seq_len_tensor, test_seq_len_tensor), dim=0)

# Ensure the input size matches the expected shape
input_size = X.shape[2]

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[range(len(output)), seq_lengths - 1, :]
        output = self.fc(output)
        return output

# Recommended hyperparameters
hidden_size = 256
num_layers = 2
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_train_losses, all_val_losses = [], []
all_mae_train, all_mae_val = [], []
all_mse_train, all_mse_val = [], []
all_r2_train, all_r2_val = [], []
all_r_train, all_r_val = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    seq_len_train, seq_len_val = seq_len[train_index], seq_len[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUModel(input_size, hidden_size, num_layers, 1).to(device)
    criterion = nn.L1Loss()  # Using MAE as the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    mae_train_scores, mae_val_scores = [], []
    mse_train_scores, mse_val_scores = [], []
    r2_train_scores, r2_val_scores = [], []
    r_train_scores, r_val_scores = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch, seq_len_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs = model(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))

        y_train_pred = model(X_train.to(device), seq_len_train.to(device)).squeeze()
        y_train_np = y_train.cpu().numpy()
        y_train_pred_np = y_train_pred.detach().cpu().numpy()  # Detach before converting to NumPy
        mae_train = mean_absolute_error(y_train_np, y_train_pred_np)
        mse_train = mean_squared_error(y_train_np, y_train_pred_np)
        r2_train = r2_score(y_train_np, y_train_pred_np)
        correlation_matrix_train = np.corrcoef(y_train_np, y_train_pred_np)
        correlation_xy_train = correlation_matrix_train[0, 1]
        r_train = correlation_xy_train

        y_val_pred = model(X_val.to(device), seq_len_val.to(device)).squeeze()
        y_val_np = y_val.cpu().numpy()
        y_val_pred_np = y_val_pred.detach().cpu().numpy()  # Detach before converting to NumPy
        mae_val = mean_absolute_error(y_val_np, y_val_pred_np)
        mse_val = mean_squared_error(y_val_np, y_val_pred_np)
        r2_val = r2_score(y_val_np, y_val_pred_np)
        correlation_matrix_val = np.corrcoef(y_val_np, y_val_pred_np)
        correlation_xy_val = correlation_matrix_val[0, 1]
        r_val = correlation_xy_val

        mae_train_scores.append(mae_train)
        mse_train_scores.append(mse_train)
        r2_train_scores.append(r2_train)
        r_train_scores.append(r_train)
        mae_val_scores.append(mae_val)
        mse_val_scores.append(mse_val)
        r2_val_scores.append(r2_val)
        r_val_scores.append(r_val)

        scheduler.step(np.mean(epoch_val_losses))

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_mae_train.append(mae_train_scores)
    all_mae_val.append(mae_val_scores)
    all_mse_train.append(mse_train_scores)
    all_mse_val.append(mse_val_scores)
    all_r2_train.append(r2_train_scores)
    all_r2_val.append(r2_val_scores)
    all_r_train.append(r_train_scores)
    all_r_val.append(r_val_scores)

# Average losses over folds for each epoch
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)
avg_mae_train = np.mean(all_mae_train, axis=0)
avg_mae_val = np.mean(all_mae_val, axis=0)
avg_r2_train = np.mean(all_r2_train, axis=0)
avg_r2_val = np.mean(all_r2_val, axis=0)

# Print averaged results for the last epoch
print(f'Average Train Loss: {avg_train_losses[-1]:.4f}')
print(f'Average Val Loss: {avg_val_losses[-1]:.4f}')
print(f'Average MAE (Train): {avg_mae_train[-1]:.4f}')
print(f'Average MSE (Train): {np.mean([mse[-1] for mse in all_mse_train]):.4f}')
print(f'Average R² (Train): {avg_r2_train[-1]:.4f}')
print(f'Average Correlation coefficient (r) (Train): {np.mean([r[-1] for r in all_r_train]):.4f}')
print(f'Average MAE (Val): {avg_mae_val[-1]:.4f}')
print(f'Average MSE (Val): {np.mean([mse[-1] for mse in all_mse_val]):.4f}')
print(f'Average R² (Val): {avg_r2_val[-1]:.4f}')
print(f'Average Correlation coefficient (r) (Val): {np.mean([r[-1] for r in all_r_val]):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'gru_model3.pth')

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_train_losses, label='Train Loss')
plt.plot(range(num_epochs), avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Plot MAE over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_mae_train, label='Train MAE')
plt.plot(range(num_epochs), avg_mae_val, label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE Over Epochs')
plt.show()

# Plot R² over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_r2_train, label='Train R²')
plt.plot(range(num_epochs), avg_r2_val, label='Validation R²')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.legend()
plt.title('R² Over Epochs')
plt.show()

# In[2]:
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Step 1: Load and prepare data (this should match your data preparation process)
# Assuming embeddings and labels are already loaded as numpy arrays

# Load embeddings and labels (your actual data loading process)
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)

with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)

with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)

with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

X = np.concatenate((longitudinal_train, longitudinal_test), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Step 2: Split data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Step 3: Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Define the GRU Model with Attention
class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUWithAttention, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_scores = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_scores.unsqueeze(-1) * gru_out, dim=1)
        output = self.fc(context_vector)
        return output, attention_scores

# Step 5: Initialize Model, Loss Function, and Optimizer
input_size = X_tensor.shape[2]
hidden_size = 256
num_layers = 2
output_size = 1

model = GRUWithAttention(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training the Model
num_epochs = 50
model.train()

for epoch in range(num_epochs):
    for X_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        outputs, attention_scores = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Step 7: Evaluate and Visualize Attention Scores
model.eval()
attention_scores_list = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs, attention_scores = model(X_batch)
        attention_scores_list.append(attention_scores.cpu().numpy())

attention_scores_combined = np.vstack(attention_scores_list)
mean_attention_scores = np.mean(attention_scores_combined, axis=0)

# Visualize the mean attention scores
plt.figure(figsize=(12, 6))
plt.bar(range(len(mean_attention_scores)), mean_attention_scores)
plt.xlabel('Window')
plt.ylabel('Mean Attention Score')
plt.title('Mean Attention Scores across all Windows')
plt.show()

# Step 8: Identify High and Low Importance Windows
high_importance_idx = np.argmax(mean_attention_scores)
low_importance_idx = np.argmin(mean_attention_scores)

print(f'High Importance Window: {high_importance_idx}, Score: {mean_attention_scores[high_importance_idx]}')
print(f'Low Importance Window: {low_importance_idx}, Score: {mean_attention_scores[low_importance_idx]}')
