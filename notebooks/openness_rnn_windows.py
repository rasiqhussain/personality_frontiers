# In[1]:
import pickle

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

# Extracting a few samples
def extract_samples(data, labels, seq_len, num_samples=3):
    samples = []
    for i in range(num_samples):
        sample_data = data[i]
        sample_label = labels[i]
        sample_seq_len = seq_len[i]
        samples.append((sample_data, sample_label, sample_seq_len))
    return samples

train_samples = extract_samples(longitudinal_train, train_labels, train_seq_len)
test_samples = extract_samples(longitudinal_test, test_labels, test_seq_len)

# Checking the format and uniformity of the embeddings
def check_format_and_uniformity(data):
    embedding_shapes = [d.shape for d in data]
    unique_shapes = set(embedding_shapes)
    return unique_shapes

train_shapes = check_format_and_uniformity(longitudinal_train)
test_shapes = check_format_and_uniformity(longitudinal_test)

# Displaying the results
print("Train Samples:")
for i, (data, label, seq_len) in enumerate(train_samples):
    print(f"Sample {i+1}:")
    print("Data shape:", data.shape)
    print("Label:", label)
    print("Sequence Length:", seq_len)
    print()

print("Test Samples:")
for i, (data, label, seq_len) in enumerate(test_samples):
    print(f"Sample {i+1}:")
    print("Data shape:", data.shape)
    print("Label:", label)
    print("Sequence Length:", seq_len)
    print()

print("Unique shapes in Train Data:", train_shapes)
print("Unique shapes in Test Data:", test_shapes)

# In[2]:
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) OR (seq_len, batch_size, d_model)
        if x.dim() == 3 and x.shape[0] == 1:
            # assume shape (1, seq_len, d_model)
            x = x + self.pe[:, :x.size(1)]
        elif x.dim() == 3 and x.shape[0] != 1:
            # if shape is (seq_len, batch_size, d_model)
            x = x + self.pe[:, :x.size(0)].transpose(0, 1)
        return self.dropout(x)

# ----------------------------
# Transformer Regressor Model
# ----------------------------
class RobertaTransformerRegressor(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=2, 
                 dim_feedforward=2048, dropout=0.1, output_size=1):
        """
        Args:
            d_model: Dimensionality of the embeddings (should match your 1024)
            nhead: Number of heads in the multihead attention mechanism
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimensionality of the feedforward network in each layer
            dropout: Dropout rate
            output_size: Final regression output size (1 for a single score)
        """
        super(RobertaTransformerRegressor, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=250)  # set max_len to cover your sequence length (e.g., 200)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x, seq_lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            seq_lengths: Tensor of actual lengths (batch_size,) to help ignore padded positions.
                         If provided, it should be an int tensor.
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create an attention mask based on seq_lengths (True for positions to be masked)
        if seq_lengths is not None:
            # Build a key_padding_mask for the transformer:
            # shape (batch_size, seq_len) with True in the positions that should be ignored (padded)
            device = x.device
            # Create a range row vector and compare to each length
            key_padding_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= seq_lengths.unsqueeze(1)
        else:
            key_padding_mask = None

        # Permute input to shape (seq_len, batch_size, d_model) as expected by nn.TransformerEncoder
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)  # add positional encoding
        # Pass through transformer encoder layers
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # Permute back to (batch_size, seq_len, d_model)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # Pool across the time dimension using a mask-aware mean pooling
        if seq_lengths is not None:
            # Create a mask of shape (batch_size, seq_len, 1): True for non-padded tokens
            mask = (~key_padding_mask).unsqueeze(2).float()
            sum_embeddings = (transformer_output * mask).sum(dim=1)
            # Avoid division by zero by converting seq_lengths to float and unsqueezing
            pooled_output = sum_embeddings / seq_lengths.unsqueeze(1).float()
        else:
            # Simple mean pooling over the sequence
            pooled_output = transformer_output.mean(dim=1)
        
        # Final regression output
        output = self.fc(pooled_output)
        return output

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import pickle
import os

# ================================
# Set random seeds for reproducibility
# ================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ================================
# Data Loading
# ================================
# Update these file paths as needed.
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)
with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

# Convert to NumPy arrays then to PyTorch tensors
X_train = np.array(longitudinal_train)  # Expected shape: (N_train, 200, 1024)
y_train = np.array(train_labels)          # Expected shape: (N_train,) or (N_train,1)
X_test = np.array(longitudinal_test)      # Expected shape: (N_test, 200, 1024)
y_test = np.array(test_labels)            # Expected shape: (N_test,) or (N_test,1)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Ensure targets are of shape (N,1)
if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)

# ================================
# Internal Train/Validation Split
# ================================
# Define a fraction of the training data to use for validation (for learning rate scheduling and monitoring)
val_fraction = 0.1
num_train_samples = len(X_train)
val_size = int(num_train_samples * val_fraction)
if val_size > 0:
    perm = torch.randperm(num_train_samples)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_actual = X_train[train_indices]
    y_train_actual = y_train[train_indices]
else:
    X_train_actual = X_train
    y_train_actual = y_train
    X_val = None
    y_val = None

# ================================
# DataLoaders
# ================================
batch_size = 32
train_dataset = TensorDataset(X_train_actual, y_train_actual)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if X_val is not None:
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
else:
    val_loader = None

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ================================
# Device configuration
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# Hyperparameters
# ================================
num_epochs = 50          # Maximum epochs
learning_rate = 1e-4      # Initial learning rate
lr_scheduler_factor = 0.5 # Factor to reduce learning rate
lr_scheduler_patience = 2 # Patience for LR scheduler

# ================================
# Initialize the Model
# ================================
# Replace 'RobertaTransformerRegressor' with your actual pre-trained model class.
model = RobertaTransformerRegressor().to(device)

# ================================
# Optimizer, Loss, and Scheduler
# ================================
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # MSE loss for training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=lr_scheduler_factor,
                                                 patience=lr_scheduler_patience,
                                                 verbose=True)

# ================================
# Training Loop (without early stopping)
# ================================
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # Forward pass
        preds = model(batch_X)
        preds = preds.squeeze(-1)
        targets = batch_y.squeeze(-1)
        
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
    
    avg_train_loss = running_loss / len(train_loader.dataset)

    # ---------------------
    # Validation Phase
    # ---------------------
    if val_loader is not None:
        model.eval()
        val_sq_error = 0.0
        val_abs_error = 0.0
        total_val_samples = 0
        y_val_true = []
        y_val_pred_list = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_X)
                preds = preds.squeeze(-1)
                targets = batch_y.squeeze(-1)
                diff = preds - targets
                val_sq_error += (diff ** 2).sum().item()
                val_abs_error += diff.abs().sum().item()
                total_val_samples += targets.size(0)
                y_val_true.extend(targets.cpu().numpy())
                y_val_pred_list.extend(preds.cpu().numpy())
        val_mse = val_sq_error / total_val_samples
        val_mae = val_abs_error / total_val_samples
        # Compute R² for validation data
        y_val_true = np.array(y_val_true)
        y_val_pred_list = np.array(y_val_pred_list)
        sst = np.sum((y_val_true - y_val_true.mean()) ** 2)
        ssr = np.sum((y_val_true - y_val_pred_list) ** 2)
        val_r2 = 1 - ssr/sst if sst != 0 else 1.0
    else:
        val_mse = avg_train_loss
        val_mae = 0.0
        val_r2 = 0.0

    print(f"Epoch {epoch:03d}: Train MSE={avg_train_loss:.4f}, Val MSE={val_mse:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")
    scheduler.step(val_mse)

# ================================
# Save the Final Model
# ================================
final_model_path = "final_model.pt"
torch.save(model.state_dict(), final_model_path)

# ================================
# Evaluate Final Model on Train and Test Sets
# ================================
# Function to calculate metrics given a dataloader
def evaluate_model(loader, model, device):
    model.eval()
    sq_error = 0.0
    abs_error = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X)
            preds = preds.squeeze(-1)
            targets = batch_y.squeeze(-1)
            diff = preds - targets
            sq_error += (diff ** 2).sum().item()
            abs_error += diff.abs().sum().item()
            total_samples += targets.size(0)
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
    mse = sq_error / total_samples
    mae = abs_error / total_samples
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    ssr = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ssr/sst if sst != 0 else 1.0
    return mse, mae, r2

# Evaluate on training data (internal training set)
train_mse, train_mae, train_r2 = evaluate_model(train_loader, model, device)
# Evaluate on test data
test_mse, test_mae, test_r2 = evaluate_model(test_loader, model, device)

print("\n=== Final Metrics ===")
print(f"Train: R²={train_r2:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}")
print(f"Test:  R²={test_r2:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")

# In[3]:
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
        if x.dim() == 3 and x.shape[0] == 1:
            # assume shape (1, seq_len, d_model)
            x = x + self.pe[:, :x.size(1)]
        elif x.dim() == 3 and x.shape[0] != 1:
            # if shape is (seq_len, batch_size, d_model)
            x = x + self.pe[:, :x.size(0)].transpose(0, 1)
        return self.dropout(x)

# ----------------------------
# Transformer Regressor Model
# ----------------------------
class RobertaTransformerRegressor(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=2, 
                 dim_feedforward=2048, dropout=0.1, output_size=1):
        """
        Args:
            d_model: Dimensionality of the embeddings (should match your 1024)
            nhead: Number of heads in the multihead attention mechanism
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimensionality of the feedforward network in each layer
            dropout: Dropout rate
            output_size: Final regression output size (1 for a single score)
        """
        super(RobertaTransformerRegressor, self).__init__()
        # Set max_len to cover your sequence length (e.g., 200)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=250)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x, seq_lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            seq_lengths: Tensor of actual lengths (batch_size,) to help ignore padded positions.
                         If provided, it should be an int tensor.
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create an attention mask based on seq_lengths (True for positions to be masked)
        if seq_lengths is not None:
            device = x.device
            key_padding_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= seq_lengths.unsqueeze(1)
        else:
            key_padding_mask = None

        # Permute input to shape (seq_len, batch_size, d_model) as expected by TransformerEncoder
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)  # Add positional encoding
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # Permute back to (batch_size, seq_len, d_model)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # Pool across the time dimension using mask-aware mean pooling if seq_lengths is provided
        if seq_lengths is not None:
            mask = (~key_padding_mask).unsqueeze(2).float()  # shape (batch_size, seq_len, 1)
            sum_embeddings = (transformer_output * mask).sum(dim=1)
            pooled_output = sum_embeddings / seq_lengths.unsqueeze(1).float()
        else:
            pooled_output = transformer_output.mean(dim=1)
        
        # Final regression output
        output = self.fc(pooled_output)
        return output

# ================================
# Data Loading (Predefined Splits)
# ================================
# The training and test data are loaded from predefined files.
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)
with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

# Convert to NumPy arrays (expected shapes are in the comments)
X_train = np.array(longitudinal_train)  # e.g., (N_train, 200, 1024)
y_train = np.array(train_labels)          # e.g., (N_train,) or (N_train,1)
X_test = np.array(longitudinal_test)      # e.g., (N_test, 200, 1024)
y_test = np.array(test_labels)            # e.g., (N_test,) or (N_test,1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Ensure that label tensors have shape (N, 1)
if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)

# ================================
# Create DataLoaders for Predefined Splits
# ================================
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ================================
# Device configuration
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# Hyperparameters
# ================================
num_epochs = 50          # Number of training epochs
learning_rate = 1e-4      # Learning rate for optimizer

# ================================
# Model, Optimizer, and Loss Function
# ================================
model = RobertaTransformerRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Regression loss

# ================================
# Training Loop (Predefined Train Split Only)
# ================================
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        preds = preds.squeeze(-1)
        targets = batch_y.squeeze(-1)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:03d}: Train MSE={avg_train_loss:.4f}")

# ================================
# Save the Trained Model
# ================================
final_model_path = "final_model.pt"
torch.save(model.state_dict(), final_model_path)

# ================================
# Evaluation Function
# ================================
def evaluate_model(loader, model, device):
    model.eval()
    sq_error = 0.0
    abs_error = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X)
            preds = preds.squeeze(-1)
            targets = batch_y.squeeze(-1)
            diff = preds - targets
            sq_error += (diff ** 2).sum().item()
            abs_error += diff.abs().sum().item()
            total_samples += targets.size(0)
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
    
    mse = sq_error / total_samples
    mae = abs_error / total_samples
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    ssr = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ssr / sst if sst != 0 else 1.0
    return mse, mae, r2

# ================================
# Final Evaluation on Predefined Training and Test Splits
# ================================
train_mse, train_mae, train_r2 = evaluate_model(train_loader, model, device)
test_mse, test_mae, test_r2 = evaluate_model(test_loader, model, device)

print("\n=== Final Evaluation Metrics ===")
print(f"Train: R²={train_r2:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}")
print(f"Test:  R²={test_r2:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")

# In[4]:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import random
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import RobertaForSequenceClassification

# -----------------------------
# Set random seeds for reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------
# Data Loading
# -----------------------------
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)   # Expected shape: (N_train, 200, 1024)
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)      # Expected shape: (N_test, 200, 1024)

with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)           # List/array of continuous targets
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)            # List/array of continuous targets

with open('train_seq_len', 'rb') as handle:
    train_seq_len = pickle.load(handle)          # Each sample: (200, 1024) with padded rows of zeros at the end
with open('test_seq_len', 'rb') as handle:
    test_seq_len = pickle.load(handle)           # Each sample: (200, 1024)

# -----------------------------
# Compute Attention Masks from the "Sequence Length" Files
# -----------------------------
# For each sample, we sum the absolute values along the embedding dimension (axis 2).
# If a row sums to 0, we treat it as padded (mask value 0); otherwise, mask value 1.
train_attention_mask = (np.sum(np.abs(train_seq_len), axis=2) > 0).astype(int)  # Resulting shape: (N_train, 200)
test_attention_mask = (np.sum(np.abs(test_seq_len), axis=2) > 0).astype(int)    # Resulting shape: (N_test, 200)

# -----------------------------
# Convert Data to PyTorch Tensors
# -----------------------------
# Our inputs are already precomputed embeddings, so we feed them directly as inputs_embeds.
train_inputs = torch.tensor(longitudinal_train, dtype=torch.float32)   # Shape: (N_train, 200, 1024)
test_inputs = torch.tensor(longitudinal_test, dtype=torch.float32)     # Shape: (N_test, 200, 1024)

train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)  # Shape: (N_train, 200)
test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)    # Shape: (N_test, 200)

# Convert labels to tensors and ensure they have shape (N,1)
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# Create TensorDatasets and DataLoaders
# -----------------------------
train_dataset = TensorDataset(train_inputs, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_inputs, test_attention_mask, test_labels)

# Optional internal train/validation split (e.g., 10% validation)
val_fraction = 0.1
num_train_samples = len(train_dataset)
val_size = int(num_train_samples * val_fraction)
indices = torch.randperm(num_train_samples)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_dataset_actual = Subset(train_dataset, train_indices)
val_dataset = Subset(train_dataset, val_indices)

batch_size = 16
train_loader = DataLoader(train_dataset_actual, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# Load Pre-trained RoBERTa for Regression
# -----------------------------
# We are using roberta-large because its hidden size (1024) matches your embeddings.
# Setting num_labels=1 makes it a regression model.
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1)
model.to(device)

# -----------------------------
# Optimizer and Loss Function
# -----------------------------
# For transformer models, AdamW is recommended.
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()  # This is used for any extra logging if desired

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 3  # Adjust as needed

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        # Unpack the batch: inputs_embeds, attention_mask, labels
        inputs_embeds, attention_mask, labels = batch
        inputs_embeds = inputs_embeds.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # Pass embeddings directly using the inputs_embeds argument.
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs_embeds.size(0)
    
    avg_train_loss = running_loss / len(train_dataset_actual)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs_embeds, attention_mask, labels = batch
            inputs_embeds = inputs_embeds.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item() * inputs_embeds.size(0)
    avg_val_loss = val_loss / len(val_dataset)
    
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

# -----------------------------
# Evaluation Function (R², MSE, MAE)
# -----------------------------
def evaluate(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs_embeds, attention_mask, labels = batch
            inputs_embeds = inputs_embeds.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            preds = outputs.logits  # Output shape: [batch_size, 1]
            all_preds.extend(preds.squeeze(-1).cpu().numpy())
            all_labels.extend(labels.squeeze(-1).cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    return mse, mae, r2

# -----------------------------
# Evaluate on the Entire Train and Test Sets
# -----------------------------
train_mse, train_mae, train_r2 = evaluate(DataLoader(train_dataset, batch_size=batch_size, shuffle=False), model, device)
test_mse, test_mae, test_r2 = evaluate(test_loader, model, device)

print("\n=== Final Metrics ===")
print(f"Train: R² = {train_r2:.4f}, MSE = {train_mse:.4f}, MAE = {train_mae:.4f}")
print(f"Test:  R² = {test_r2:.4f}, MSE = {test_mse:.4f}, MAE = {test_mae:.4f}")

# -----------------------------
# Save the Final Model
# -----------------------------
model.save_pretrained("final_roberta_regressor")

# In[5]:
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
torch.save(model.state_dict(), 'gru_model.pth')

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

# In[6]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
X_train_tensor = torch.tensor(longitudinal_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(longitudinal_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(train_labels, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)

train_seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_train], dtype=torch.int64).to(device)
test_seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_test], dtype=torch.int64).to(device)

# Define the GRU model with Attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * output, dim=1)

        output = self.fc(context_vector)
        return output, attention_weights

# Model parameters
input_size = X_train_tensor.shape[2]
hidden_size = 256
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Initialize model, loss function, optimizer
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Prepare DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_seq_len_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop with TQDM progress bar
train_losses = []
val_losses = []
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch, seq_len_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
        optimizer.zero_grad()
        outputs, _ = model(X_batch, seq_len_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, seq_len_batch in test_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(test_loader)
    val_losses.append(epoch_val_loss)

    print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'gru_attention_model.pth')

# Separate module for visualization
def visualize_attention(model, dataloader, top_n=5):
    model.eval()
    top_windows = []
    bottom_windows = []
    with torch.no_grad():
        for X_batch, y_batch, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            avg_attention = attention_weights.mean(dim=0).cpu().numpy()
            top_windows.extend(np.argsort(-avg_attention)[:top_n])
            bottom_windows.extend(np.argsort(avg_attention)[:top_n])
    
    # Plot attention distribution
    plt.figure(figsize=(10, 5))
    sns.heatmap([avg_attention], cmap='viridis')
    plt.title('Attention Distribution Across Windows')
    plt.xlabel('Window Index')
    plt.ylabel('Attention Score')
    plt.show()

# Run visualization after training
visualize_attention(model, test_loader)

# In[7]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for test set
attention_scores = calculate_attention_scores(model, test_loader)

# Analyze top and bottom 5 windows for each transcript
def analyze_windows(transcripts, attention_scores, top_n=5):
    top_windows = []
    bottom_windows = []
    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_n:][::-1]
        bottom_indices = sorted_indices[:top_n]
        top_windows.append((transcript['PARTID'], top_indices))
        bottom_windows.append((transcript['PARTID'], bottom_indices))
    return top_windows, bottom_windows

top_windows, bottom_windows = analyze_windows(test_transcripts, attention_scores)

# Function to visualize attention scores and transcripts
def visualize_attention_scores(transcripts, attention_scores, top_windows, bottom_windows):
    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        plt.figure(figsize=(10, 4))
        sns.heatmap([scores], cmap='viridis', cbar=True)
        plt.title(f"Attention Scores for Transcript ID: {transcript['PARTID']}")
        plt.xlabel('Window Index')
        plt.ylabel('Attention Score')
        plt.show()
        print(f"Transcript ID: {transcript['PARTID']}")
        print("Top 5 windows contributing to the score:", top_windows[i][1])
        print("Bottom 5 windows contributing to the score:", bottom_windows[i][1])
        print("\n")

# Visualize and interpret the attention scores
visualize_attention_scores(test_transcripts, attention_scores, top_windows, bottom_windows)

# In[8]:
# Define the target transcript ID
target_transcript_id = 4247

# Initialize a variable to hold the attention weights
target_attention_weights = None

# Iterate over transcripts and their corresponding attention scores
for transcript, attn_weights in zip(test_transcripts, attention_scores):
    # Convert PARTID to int if stored as a string. Adjust accordingly if needed.
    if int(transcript['PARTID']) == target_transcript_id:
        target_attention_weights = attn_weights
        break

# Check if the transcript was found and save the weights
if target_attention_weights is None:
    print(f"No transcript with PARTID {target_transcript_id} found.")
else:
    # Save the attention weights to a NumPy file
    np.save('attention_weights_transcript_4247.npy', target_attention_weights)
    print(f"Attention weights for transcript {target_transcript_id} saved to 'attention_weights_transcript_4247.npy'.")

# In[9]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for test set
attention_scores = calculate_attention_scores(model, test_loader)

# Analyze top and bottom 5 windows for each transcript and divide windows into "early," "middle," "late"
def analyze_windows(transcripts, attention_scores, top_n=5):
    top_windows = []
    bottom_windows = []
    section_contributions = {'early': [], 'middle': [], 'late': []}

    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_n:][::-1]
        bottom_indices = sorted_indices[:top_n]
        top_windows.append((transcript['PARTID'], top_indices))
        bottom_windows.append((transcript['PARTID'], bottom_indices))
        
        # Determine the number of windows and divide into early, middle, late
        num_windows = len(scores)
        third = num_windows // 3
        remainder = num_windows % 3

        early_indices = range(0, third)
        middle_indices = range(third, 2*third + remainder)  # middle gets the extra if remainder exists
        late_indices = range(2*third + remainder, num_windows)

        # Calculate average attention score for each section
        early_avg = np.mean(scores[early_indices])
        middle_avg = np.mean(scores[middle_indices])
        late_avg = np.mean(scores[late_indices])

        section_contributions['early'].append(early_avg)
        section_contributions['middle'].append(middle_avg)
        section_contributions['late'].append(late_avg)

    return top_windows, bottom_windows, section_contributions

top_windows, bottom_windows, section_contributions = analyze_windows(test_transcripts, attention_scores)

# Function to visualize attention scores and transcripts with sections
def visualize_attention_scores(transcripts, attention_scores, top_windows, bottom_windows, section_contributions):
    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        plt.figure(figsize=(12, 5))
        num_windows = len(scores)
        third = num_windows // 3
        remainder = num_windows % 3

        # Create sections for visualization
        early_indices = range(0, third)
        middle_indices = range(third, 2*third + remainder)
        late_indices = range(2*third + remainder, num_windows)

        plt.plot(range(num_windows), scores, label='Attention Scores', color='blue')
        plt.axvspan(0, third, color='red', alpha=0.3, label='Early')  # Early section
        plt.axvspan(third, 2*third + remainder, color='green', alpha=0.3, label='Middle')  # Middle section
        plt.axvspan(2*third + remainder, num_windows, color='yellow', alpha=0.3, label='Late')  # Late section

        plt.title(f"Attention Scores for Transcript ID: {transcript['PARTID']}")
        plt.xlabel('Window Index')
        plt.ylabel('Attention Score')
        plt.legend()
        plt.show()

        print(f"Transcript ID: {transcript['PARTID']}")
        print("Top 5 windows contributing to the score:", top_windows[i][1])
        print("Bottom 5 windows contributing to the score:", bottom_windows[i][1])
        print(f"Average Attention Scores - Early: {section_contributions['early'][i]:.4f}, "
              f"Middle: {section_contributions['middle'][i]:.4f}, "
              f"Late: {section_contributions['late'][i]:.4f}")
        print("\n")

# Visualize and interpret the attention scores
visualize_attention_scores(test_transcripts, attention_scores, top_windows, bottom_windows, section_contributions)

# In[10]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for test set
attention_scores = calculate_attention_scores(model, test_loader)

# Function to divide windows into early, middle, and late
def divide_windows(scores):
    num_windows = len(scores)
    third = num_windows // 3
    if num_windows % 3 == 0:
        early = scores[:third]
        middle = scores[third:2*third]
        late = scores[2*third:]
    else:
        early = scores[:third]
        middle = scores[third:2*third]
        late = scores[2*third:]
    return early, middle, late

# Analyze the contributions of each section across all transcripts
def analyze_contributions(transcripts, attention_scores):
    early_contributions = []
    middle_contributions = []
    late_contributions = []
    
    for scores in attention_scores:
        early, middle, late = divide_windows(scores)
        early_contributions.append(np.mean(early))
        middle_contributions.append(np.mean(middle))
        late_contributions.append(np.mean(late))
    
    avg_early = np.mean(early_contributions)
    avg_middle = np.mean(middle_contributions)
    avg_late = np.mean(late_contributions)
    
    print(f"Average contribution of Early windows: {avg_early:.4f}")
    print(f"Average contribution of Middle windows: {avg_middle:.4f}")
    print(f"Average contribution of Late windows: {avg_late:.4f}")
    
    return avg_early, avg_middle, avg_late

# Analyze top and bottom 5 windows for each transcript
def analyze_windows(transcripts, attention_scores, top_n=5):
    top_windows = []
    bottom_windows = []
    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_n:][::-1]
        bottom_indices = sorted_indices[:top_n]
        top_windows.append((transcript['PARTID'], top_indices))
        bottom_windows.append((transcript['PARTID'], bottom_indices))
    return top_windows, bottom_windows

# Function to visualize attention scores and transcript sections
def visualize_attention_scores(transcripts, attention_scores, avg_early, avg_middle, avg_late):
    for i, (transcript, scores) in enumerate(zip(transcripts, attention_scores)):
        plt.figure(figsize=(10, 4))
        sns.heatmap([scores], cmap='viridis', cbar=True)
        plt.title(f"Attention Scores for Transcript ID: {transcript['PARTID']}")
        plt.xlabel('Window Index')
        plt.ylabel('Attention Score')
        plt.show()
        
        # Highlight contributions of each section
        early, middle, late = divide_windows(scores)
        print(f"Transcript ID: {transcript['PARTID']}")
        print("Contribution of Early windows:", np.mean(early))
        print("Contribution of Middle windows:", np.mean(middle))
        print("Contribution of Late windows:", np.mean(late))
        print("\n")
    
    # Bar plot for overall contribution comparison
    plt.figure(figsize=(8, 5))
    sections = ['Early', 'Middle', 'Late']
    contributions = [avg_early, avg_middle, avg_late]
    plt.bar(sections, contributions, color=['blue', 'orange', 'green'])
    plt.title('Average Contributions of Different Sections')
    plt.xlabel('Section')
    plt.ylabel('Average Attention Score')
    plt.show()

# Calculate and visualize contributions
avg_early, avg_middle, avg_late = analyze_contributions(test_transcripts, attention_scores)
visualize_attention_scores(test_transcripts, attention_scores, avg_early, avg_middle, avg_late)

# In[11]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_seq_len_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for both train and test sets
train_attention_scores = calculate_attention_scores(model, train_loader)
test_attention_scores = calculate_attention_scores(model, test_loader)

# Function to divide windows into early, middle, and late
def divide_windows(scores):
    num_windows = len(scores)
    third = num_windows // 3
    early = scores[:third]
    middle = scores[third:2*third]
    late = scores[2*third:]
    return early, middle, late

# Analyze the contributions of each section across all embeddings
def analyze_contributions(attention_scores):
    early_contributions = []
    middle_contributions = []
    late_contributions = []
    
    for scores in attention_scores:
        early, middle, late = divide_windows(scores)
        early_contributions.append(np.mean(early))
        middle_contributions.append(np.mean(middle))
        late_contributions.append(np.mean(late))
    
    avg_early = np.mean(early_contributions)
    avg_middle = np.mean(middle_contributions)
    avg_late = np.mean(late_contributions)
    
    print(f"Average contribution of Early windows: {avg_early:.4f}")
    print(f"Average contribution of Middle windows: {avg_middle:.4f}")
    print(f"Average contribution of Late windows: {avg_late:.4f}")
    
    return avg_early, avg_middle, avg_late

# Analyze top and bottom 5 windows for each embedding
def analyze_windows(attention_scores, top_n=5):
    top_windows = []
    bottom_windows = []
    for i, scores in enumerate(attention_scores):
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_n:][::-1]
        bottom_indices = sorted_indices[:top_n]
        top_windows.append((i, top_indices))
        bottom_windows.append((i, bottom_indices))
    return top_windows, bottom_windows

# Function to visualize attention scores and embeddings sections
def visualize_attention_scores(attention_scores, avg_early, avg_middle, avg_late):
    for i, scores in enumerate(attention_scores):
        plt.figure(figsize=(10, 4))
        sns.heatmap([scores], cmap='viridis', cbar=True)
        plt.title(f"Attention Scores for Embedding Index: {i}")
        plt.xlabel('Window Index')
        plt.ylabel('Attention Score')
        plt.show()
        
        # Highlight contributions of each section
        early, middle, late = divide_windows(scores)
        print(f"Embedding Index: {i}")
        print("Contribution of Early windows:", np.mean(early))
        print("Contribution of Middle windows:", np.mean(middle))
        print("Contribution of Late windows:", np.mean(late))
        print("\n")
    
    # Bar plot for overall contribution comparison
    plt.figure(figsize=(8, 5))
    sections = ['Early', 'Middle', 'Late']
    contributions = [avg_early, avg_middle, avg_late]
    plt.bar(sections, contributions, color=['blue', 'orange', 'green'])
    plt.title('Average Contributions of Different Sections')
    plt.xlabel('Section')
    plt.ylabel('Average Attention Score')
    plt.show()

# Calculate and visualize contributions for both train and test sets
print("Train Set Analysis:")
avg_early_train, avg_middle_train, avg_late_train = analyze_contributions(train_attention_scores)
visualize_attention_scores(train_attention_scores, avg_early_train, avg_middle_train, avg_late_train)

print("Test Set Analysis:")
avg_early_test, avg_middle_test, avg_late_test = analyze_contributions(test_attention_scores)
visualize_attention_scores(test_attention_scores, avg_early_test, avg_middle_test, avg_late_test)

# In[12]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Function to divide windows into early, middle, and late
def divide_windows(scores):
    num_windows = len(scores)
    third = num_windows // 3
    if num_windows % 3 == 0:
        early = scores[:third]
        middle = scores[third:2*third]
        late = scores[2*third:]
    else:
        early = scores[:third]
        middle = scores[third:2*third]
        late = scores[2*third:]
    return early, middle, late

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for test set
attention_scores = calculate_attention_scores(model, test_loader)

# Extract E scores and sort transcripts based on these scores
E_scores = [transcript["PNEOO_scaled"] for transcript in test_transcripts]
sorted_indices = np.argsort(E_scores)

# Get top 5 and bottom 5 transcripts based on E scores
top_5_indices = sorted_indices[-5:]
bottom_5_indices = sorted_indices[:5]

top_5_transcripts = [test_transcripts[i] for i in top_5_indices]
bottom_5_transcripts = [test_transcripts[i] for i in bottom_5_indices]

top_5_attention_scores = [attention_scores[i] for i in top_5_indices]
bottom_5_attention_scores = [attention_scores[i] for i in bottom_5_indices]

# Get PARTIDs of top and bottom transcripts
top_5_PARTIDs = [transcript['PARTID'] for transcript in top_5_transcripts]
bottom_5_PARTIDs = [transcript['PARTID'] for transcript in bottom_5_transcripts]

print("Top 5 PARTIDs (Highest O Scores):", top_5_PARTIDs)
print("Bottom 5 PARTIDs (Lowest O Scores):", bottom_5_PARTIDs)

# Analyze and visualize contributions of each section
def analyze_contributions(attention_scores):
    contributions = {'early': [], 'middle': [], 'late': []}
    for scores in attention_scores:
        early, middle, late = divide_windows(scores)
        contributions['early'].append(np.mean(early))
        contributions['middle'].append(np.mean(middle))
        contributions['late'].append(np.mean(late))
    
    avg_early = np.mean(contributions['early'])
    avg_middle = np.mean(contributions['middle'])
    avg_late = np.mean(contributions['late'])
    
    return avg_early, avg_middle, avg_late

avg_early_top, avg_middle_top, avg_late_top = analyze_contributions(top_5_attention_scores)
avg_early_bottom, avg_middle_bottom, avg_late_bottom = analyze_contributions(bottom_5_attention_scores)

# Visualize comparison
def visualize_comparison(avg_top, avg_bottom):
    labels = ['Early', 'Middle', 'Late']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, avg_top, width, label='Top 5 Transcripts')
    bars2 = ax.bar(x + width/2, avg_bottom, width, label='Bottom 5 Transcripts')

    ax.set_xlabel('Section')
    ax.set_ylabel('Average Attention Score')
    ax.set_title('Average Attention Scores by Section for Top and Bottom Transcripts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Visualize results
visualize_comparison([avg_early_top, avg_middle_top, avg_late_top], [avg_early_bottom, avg_middle_bottom, avg_late_bottom])

# In[13]:
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Function to calculate attention scores using the trained model
def calculate_attention_scores(model, dataloader):
    model.eval()
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            _, attention_weights = model(X_batch, seq_len_batch)
            attention_scores.extend(attention_weights.cpu().numpy())
    return attention_scores

# Function to divide windows into early, middle, and late
def divide_windows(scores):
    num_windows = len(scores)
    third = num_windows // 3
    early = scores[:third]
    middle = scores[third:2*third]
    late = scores[2*third:]
    return early, middle, late

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth'))
model.eval()

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate attention scores for test set
attention_scores = calculate_attention_scores(model, test_loader)

# Extract O scores and sort transcripts based on these scores
E_scores = [transcript["PNEOO_scaled"] for transcript in test_transcripts]
sorted_indices = np.argsort(E_scores)

# Get top 5 and bottom 5 transcripts based on E scores
top_5_indices = sorted_indices[-5:]
bottom_5_indices = sorted_indices[:5]

top_5_transcripts = [test_transcripts[i] for i in top_5_indices]
bottom_5_transcripts = [test_transcripts[i] for i in bottom_5_indices]

top_5_attention_scores = [attention_scores[i] for i in top_5_indices]
bottom_5_attention_scores = [attention_scores[i] for i in bottom_5_indices]

# Get PARTIDs and text lengths of top and bottom transcripts
top_5_info = [(transcript['PARTID'], transcript['text length']) for transcript in top_5_transcripts]
bottom_5_info = [(transcript['PARTID'], transcript['text length']) for transcript in bottom_5_transcripts]

print("Top 5 PARTIDs and Text Lengths (Highest E Scores):", top_5_info)
print("Bottom 5 PARTIDs and Text Lengths (Lowest E Scores):", bottom_5_info)

# Analyze and visualize contributions of each section
def analyze_contributions(attention_scores):
    contributions = {'early': [], 'middle': [], 'late': []}
    for scores in attention_scores:
        early, middle, late = divide_windows(scores)
        contributions['early'].append(np.mean(early))
        contributions['middle'].append(np.mean(middle))
        contributions['late'].append(np.mean(late))
    
    avg_early = np.mean(contributions['early'])
    avg_middle = np.mean(contributions['middle'])
    avg_late = np.mean(contributions['late'])
    
    return avg_early, avg_middle, avg_late

avg_early_top, avg_middle_top, avg_late_top = analyze_contributions(top_5_attention_scores)
avg_early_bottom, avg_middle_bottom, avg_late_bottom = analyze_contributions(bottom_5_attention_scores)

# Visualize comparison
def visualize_comparison(avg_top, avg_bottom):
    labels = ['Early', 'Middle', 'Late']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, avg_top, width, label='Top 5 Transcripts')
    bars2 = ax.bar(x + width/2, avg_bottom, width, label='Bottom 5 Transcripts')

    ax.set_xlabel('Section')
    ax.set_ylabel('Average Attention Score')
    ax.set_title('Average Attention Scores by Section for Top and Bottom Transcripts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Visualize individual attention score distributions for top and bottom transcripts
def visualize_individual_attention(transcripts, attention_scores, title_prefix):
    for transcript, scores in zip(transcripts, attention_scores):
        plt.figure(figsize=(10, 4))
        sns.heatmap([scores], cmap='viridis', cbar=True)
        plt.title(f"{title_prefix} Attention Scores for Transcript ID: {transcript['PARTID']}")
        plt.xlabel('Window Index')
        plt.ylabel('Attention Score')
        plt.show()
        
        # Display contributions for each section
        early, middle, late = divide_windows(scores)
        print(f"Transcript ID: {transcript['PARTID']}")
        print("Contribution of Early windows:", np.mean(early))
        print("Contribution of Middle windows:", np.mean(middle))
        print("Contribution of Late windows:", np.mean(late))
        print("\n")

# Visualize results
visualize_comparison([avg_early_top, avg_middle_top, avg_late_top], [avg_early_bottom, avg_middle_bottom, avg_late_bottom])
visualize_individual_attention(top_5_transcripts, top_5_attention_scores, "Top 5")
visualize_individual_attention(bottom_5_transcripts, bottom_5_attention_scores, "Bottom 5")

# In[14]:
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load the Emotion Lexicon CSV
emotion_lexicon_path = 'Emotion_Lexicon.csv'
emotion_lexicon = pd.read_csv(emotion_lexicon_path)

# Create sets for positive and negative words
positive_words = set(emotion_lexicon[emotion_lexicon['positive'] == 1]['Words'])
negative_words = set(emotion_lexicon[emotion_lexicon['negative'] == 1]['Words'])

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
train_files = ['1.json', '2.json', '3.json', '4.json']
test_file = ['5.json']

train_transcripts = load_transcripts(train_files)
test_transcripts = load_transcripts(test_file)

# Filter out PARTIDs with missing scores
missing_train_ids = {2119, 2146, 2175, 2182, 2183, 2243, 2271, 2310, 2314, 2368, 2372, 2376, 2403, 3216, 3314, 3422, 3438, 3517, 4016, 4023, 4055, 4105, 4122, 4135, 4167, 4171}
missing_test_ids = {4234, 4274, 4302, 4367, 4412, 4415, 4420, 4501}

filtered_train_transcripts = [t for t in train_transcripts if t['PARTID'] not in missing_train_ids]
filtered_test_transcripts = [t for t in test_transcripts if t['PARTID'] not in missing_test_ids]

# Load the embeddings and actual Neuroticism labels for the train and test sets
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

# Load the trained model
input_size = X_train_tensor.shape[2]
hidden_size = 256
num_layers = 2
output_size = 1  # Predicting Neuroticism
model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()

# Function to calculate predictions using the trained model
def calculate_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in dataloader:
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            outputs = model(X_batch, seq_len_batch)
            predictions.extend(outputs.cpu().numpy())
    return predictions

# Prepare DataLoader for train and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_seq_len_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate predictions for train and test sets
train_predictions = calculate_predictions(model, train_loader)
train_predictions = np.array(train_predictions).flatten()

test_predictions = calculate_predictions(model, test_loader)
test_predictions = np.array(test_predictions).flatten()

# Sort the indices based on predictions
sorted_train_indices_predicted = np.argsort(train_predictions)
sorted_test_indices_predicted = np.argsort(test_predictions)

# Get top 5 and bottom 5 transcripts based on predicted N scores for train and test data
top_5_train_predicted = sorted_train_indices_predicted[-5:]
bottom_5_train_predicted = sorted_train_indices_predicted[:5]

top_5_test_predicted = sorted_test_indices_predicted[-5:]
bottom_5_test_predicted = sorted_test_indices_predicted[:5]

# Function to tokenize text and count positive/negative words
def count_emotional_words(transcripts, positive_words, negative_words):
    results = []
    for transcript in transcripts:
        text = transcript['text'].lower()
        words = re.findall(r'\b\w+\b', text)
        word_counts = Counter(words)
        
        positive_count = sum(word_counts[word] for word in positive_words if word in word_counts)
        negative_count = sum(word_counts[word] for word in negative_words if word in word_counts)
        
        results.append({
            'PARTID': transcript['PARTID'],
            'positive_count': positive_count,
            'negative_count': negative_count
        })
    return results

# Get word counts for top 5 and bottom 5 predicted test transcripts
top_5_test_transcripts_predicted = [filtered_test_transcripts[i] for i in top_5_test_predicted]
bottom_5_test_transcripts_predicted = [filtered_test_transcripts[i] for i in bottom_5_test_predicted]

top_5_word_counts_predicted = count_emotional_words(top_5_test_transcripts_predicted, positive_words, negative_words)
bottom_5_word_counts_predicted = count_emotional_words(bottom_5_test_transcripts_predicted, positive_words, negative_words)

# Display word counts for top 5 predicted transcripts
print("Top 5 Test Transcripts (Predicted O Scores) Word Counts:")
for result in top_5_word_counts_predicted:
    print(f"PARTID {result['PARTID']}: Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")

# Display word counts for bottom 5 predicted transcripts
print("\nBottom 5 Test Transcripts (Predicted O Scores) Word Counts:")
for result in bottom_5_word_counts_predicted:
    print(f"PARTID {result['PARTID']}: Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")

# In[15]:
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the Emotion Lexicon CSV
emotion_lexicon_path = 'Emotion_Lexicon.csv'
emotion_lexicon = pd.read_csv(emotion_lexicon_path)

# Create sets for positive and negative words
positive_words = set(emotion_lexicon[emotion_lexicon['positive'] == 1]['Words'])
negative_words = set(emotion_lexicon[emotion_lexicon['negative'] == 1]['Words'])

# Load JSON transcripts
def load_transcripts(file_paths):
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transcripts.extend(data)
    return transcripts

# File paths for JSON files
test_file = ['5.json']

test_transcripts = load_transcripts(test_file)

# Filter out PARTIDs with missing scores
missing_test_ids = {4234, 4274, 4302, 4367, 4412, 4415, 4420, 4501}
filtered_test_transcripts = [t for t in test_transcripts if t['PARTID'] not in missing_test_ids]

# Load the embeddings and actual Neuroticism labels for the test set
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)

with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

with open('test_seq_len', 'rb') as handle:
    test_seq_len = pickle.load(handle)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(longitudinal_test, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.float32)
test_seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_test], dtype=torch.int64)

# Define the GRU model with attention mechanism
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Correct the name here
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Calculate attention scores
        attn_scores = self.attention(output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Softmax to get window contributions
        
        # Weighted sum of outputs
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        output = self.fc(context)
        return output, attn_weights

# Load the trained model
input_size = X_test_tensor.shape[2]
hidden_size = 256
num_layers = 2
output_size = 1  # Predicting Neuroticism
model = GRUAttentionModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_attention_model.pth', map_location=device))
model.eval()

# Function to calculate predictions and attention scores
def calculate_predictions_and_attention(model, dataloader):
    model.eval()
    predictions = []
    attention_scores = []
    with torch.no_grad():
        for X_batch, _, seq_len_batch in tqdm(dataloader, desc="Processing"):
            X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
            outputs, attn_weights = model(X_batch, seq_len_batch)
            predictions.extend(outputs.cpu().numpy())
            attention_scores.extend(attn_weights.cpu().numpy())
    return predictions, attention_scores

# Prepare DataLoader for test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_seq_len_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate predictions and attention scores for the test set
test_predictions, test_attention_scores = calculate_predictions_and_attention(model, test_loader)
test_predictions = np.array(test_predictions).flatten()

# Sort the indices based on predictions
sorted_test_indices_predicted = np.argsort(test_predictions)

# Get top 5 and bottom 5 transcripts based on predicted N scores
top_5_test_predicted = sorted_test_indices_predicted[-5:]
bottom_5_test_predicted = sorted_test_indices_predicted[:5]

# Analyze the most contributing window and words within that window
def analyze_window_contributions(attention_scores, transcripts, top=True):
    results = []
    for idx in (top_5_test_predicted if top else bottom_5_test_predicted):
        attn = attention_scores[idx]
        transcript = transcripts[idx]
        most_contrib_window_idx = np.argmax(attn)  # Index of the most contributing window
        words = transcript['text'].split()  # Example word splitting, adjust as needed
        
        # Find the most contributing words
        window_size = len(words) // len(attn)
        start_idx = most_contrib_window_idx * window_size
        end_idx = start_idx + window_size
        most_contrib_words = words[start_idx:end_idx]
        
        # Count positive/negative words
        word_counts = Counter(most_contrib_words)
        positive_count = sum(word_counts[word] for word in positive_words if word in word_counts)
        negative_count = sum(word_counts[word] for word in negative_words if word in word_counts)
        
        results.append({
            'PARTID': transcript['PARTID'],
            'attention_score': attn[most_contrib_window_idx],
            'most_contrib_words': most_contrib_words,
            'positive_count': positive_count,
            'negative_count': negative_count
        })
    return results

# Get word contributions for top 5 and bottom 5 predicted test transcripts
top_5_word_contributions = analyze_window_contributions(test_attention_scores, filtered_test_transcripts, top=True)
bottom_5_word_contributions = analyze_window_contributions(test_attention_scores, filtered_test_transcripts, top=False)

# Display attention scores and most contributing words
print("Top 5 Test Transcripts (Predicted O Scores) Word Contributions:")
for result in top_5_word_contributions:
    print(f"PARTID {result['PARTID']}: Attention Score: {result['attention_score']}, Most Contributing Words: {result['most_contrib_words']}, Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")

print("\nBottom 5 Test Transcripts (Predicted O Scores) Word Contributions:")
for result in bottom_5_word_contributions:
    print(f"PARTID {result['PARTID']}: Attention Score: {result['attention_score']}, Most Contributing Words: {result['most_contrib_words']}, Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")

# In[16]:
# Function to analyze window contributions and select most contributing words
def analyze_window_contributions_by_attention(attention_scores, transcripts, indices):
    results = []
    transcript_count = len(transcripts)  # Total number of transcripts available
    for idx in indices:
        if idx >= transcript_count:  # Check if the index is valid
            print(f"Warning: Index {idx} is out of range for transcripts with length {transcript_count}")
            continue  # Skip the out-of-range index
        
        attn = attention_scores[idx]
        transcript = transcripts[idx]
        most_contrib_window_idx = np.argmax(attn)  # Index of the most contributing window
        words = transcript['text'].split()  # Example word splitting, adjust as needed
        
        # Find the most contributing words in the window
        window_size = len(words) // len(attn)
        start_idx = most_contrib_window_idx * window_size
        end_idx = start_idx + window_size
        most_contrib_words = words[start_idx:end_idx]
        
        # Count positive/negative words
        word_counts = Counter(most_contrib_words)
        positive_count = sum(word_counts[word] for word in positive_words if word in word_counts)
        negative_count = sum(word_counts[word] for word in negative_words if word in word_counts)
        
        results.append({a
            'PARTID': transcript['PARTID'],
            'attention_score': np.mean(attn),
            'most_contrib_words': most_contrib_words,
            'positive_count': positive_count,
            'negative_count': negative_count
        })
    return results

# Analyze contributions for top 5 and bottom 5 based on attention scores
top_5_contributions_by_attention = analyze_window_contributions_by_attention(test_attention_scores, filtered_test_transcripts, top_5_attention_idx)
bottom_5_contributions_by_attention = analyze_window_contributions_by_attention(test_attention_scores, filtered_test_transcripts, bottom_5_attention_idx)

# Display results
print("\nTop 5 Test Transcripts (By Attention Scores) Word Contributions:")
for result in top_5_contributions_by_attention:
    print(f"PARTID {result['PARTID']}: Attention Score: {result['attention_score']}, Most Contributing Words: {result['most_contrib_words']}, Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")

print("\nBottom 5 Test Transcripts (By Attention Scores) Word Contributions:")
for result in bottom_5_contributions_by_attention:
    print(f"PARTID {result['PARTID']}: Attention Score: {result['attention_score']}, Most Contributing Words: {result['most_contrib_words']}, Positive Words: {result['positive_count']}, Negative Words: {result['negative_count']}")
