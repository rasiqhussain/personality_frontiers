# In[1]:
# Load embeddings, labels, and sequence lengths
import pickle

with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)
with open('labels', 'rb') as handle:
    labels = pickle.load(handle)
with open('train_seq_len', 'rb') as handle:
    train_seq_len = pickle.load(handle)

# Display loaded data
print("Longitudinal Train Sample:", longitudinal_train[:2])  # Display first 2 samples for brevity
print("Labels Sample:", labels[:2])  # Display first 2 labels
print("Train Seq Length Sample:", train_seq_len[:2])  # Display first 2 sequence lengths

# In[2]:
# Display additional samples to get a better understanding
print("Additional Longitudinal Train Samples (first 3):")
for i, sample in enumerate(longitudinal_train[:3]):
    print(f"Sample {i+1} shape: {sample.shape}")
    print(sample)
    print()  # Blank line for readability

print("\nAdditional Labels Samples (first 3):")
print(labels.head(3))  # Display the first 3 rows

print("\nTrain Seq Lengths (first 3):")
print(train_seq_len[:3])  # Display the first 3 sequence lengths

# Check shapes of loaded variables
print("\nShapes of Loaded Variables:")
print("Longitudinal Train shape (overall structure):", len(longitudinal_train), "sequences with individual shapes varying per sequence.")
print("Labels shape:", labels.shape)
print("Train Seq Length shape:", train_seq_len.shape)

# In[3]:
import numpy as np

# Sample extraction for display
sample_indices = [0, 1, 2]  # Extract the first three samples
samples = {
    "Longitudinal Train Samples": [longitudinal_train[i] for i in sample_indices],
    "Labels Samples": labels.iloc[sample_indices],
    "Train Seq Lengths Samples": [train_seq_len[i] for i in sample_indices],
}

# Display sample content
print("Extracted Samples:")
for key, sample in samples.items():
    print(f"\n{key}:\n", sample)

# Check embedding uniformity
# 1. Confirm that all embeddings have shape (200, 1024)
embedding_shapes = [embedding.shape for embedding in longitudinal_train]
uniform_shape = all(shape == (200, 1024) for shape in embedding_shapes)
print("\nAre all embeddings of uniform shape (200, 1024)?", uniform_shape)

# 2. Verify that padded zeros match sequence lengths in train_seq_len
padding_checks = []
for i, (embedding, seq_len) in enumerate(zip(longitudinal_train, train_seq_len)):
    if not np.all(embedding[seq_len:] == 0):
        padding_checks.append(i)

if not padding_checks:
    print("\nAll embeddings have correct zero-padding after the sequence length.")
else:
    print("\nPadding errors found in embeddings at indices:", padding_checks)

# In[4]:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle

# Load data
with open('longitudinal_train', 'rb') as handle:
    longitudinal_train = pickle.load(handle)
with open('labels', 'rb') as handle:
    labels = pickle.load(handle)
with open('train_seq_len', 'rb') as handle:
    train_seq_len = pickle.load(handle)

# Convert to PyTorch tensors
X_tensor = torch.tensor(longitudinal_train, dtype=torch.float32)  # Shape: (num_samples, seq_length, embedding_dim)
y_tensor = torch.tensor(labels['labels'].values, dtype=torch.float32)  # Extract 'labels' column, Shape: (num_samples,)
seq_len_tensor = torch.tensor(train_seq_len, dtype=torch.long)  # Shape: (num_samples,)

# Print tensor shapes for verification
print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)
print("seq_len_tensor shape:", seq_len_tensor.shape)

# Define the GRUAttentionModel
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, attention_heads, dropout_rate, output_size):
        super(GRUAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_heads = nn.ModuleList([nn.Linear(attention_dim, 1, bias=False) for _ in range(attention_heads)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Multi-head attention computation
        attention_weights = [torch.tanh(self.attention(output)) for _ in self.attention_heads]
        attention_scores = [head(attention).squeeze(-1) for attention, head in zip(attention_weights, self.attention_heads)]
        attention_scores = torch.stack(attention_scores, dim=-1)
        attention_scores = torch.softmax(attention_scores.mean(dim=-1), dim=1)

        # Weighted context vector
        context_vector = torch.sum(attention_scores.unsqueeze(-1) * output, dim=1)
        return self.fc(self.dropout(context_vector)), attention_scores

# Training function
def train_model(hyperparams, X_tensor, y_tensor, seq_len_tensor, device="cuda"):
    # Unpack hyperparameters
    attention_dim = hyperparams['attention_dim']
    hidden_size = hyperparams['hidden_size']
    num_layers = hyperparams['num_layers']
    attention_heads = hyperparams['attention_heads']
    dropout_rate = hyperparams['dropout_rate']
    weight_decay = hyperparams['weight_decay']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    split_ratio = hyperparams['split_ratio']

    # Data split
    split_point = int(len(X_tensor) * split_ratio)
    X_train, X_val = X_tensor[:split_point], X_tensor[split_point:]
    y_train, y_val = y_tensor[:split_point], y_tensor[split_point:]
    seq_len_train, seq_len_val = seq_len_tensor[:split_point], seq_len_tensor[split_point:]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model and optimizer setup
    model = GRUAttentionModel(
        input_size=X_tensor.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        output_size=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    best_val_r2 = -np.inf
    num_epochs = 100
    early_stopping_patience = 15
    patience_counter = 0

    avg_r2_train, avg_mse_train = [], []
    avg_r2_val, avg_mse_val = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        y_train_pred, y_train_true = [], []
        epoch_train_loss = 0
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            y_train_pred.extend(outputs.view(-1).detach().cpu().numpy())
            y_train_true.extend(y_batch.detach().cpu().numpy())

        r2_train = r2_score(y_train_true, y_train_pred)
        mse_train = mean_squared_error(y_train_true, y_train_pred)
        avg_r2_train.append(r2_train)
        avg_mse_train.append(mse_train)

        # Validation phase
        model.eval()
        y_val_pred, y_val_true = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model(X_batch, seq_len_batch)
                loss = criterion(outputs.view(-1), y_batch)
                epoch_val_loss += loss.item()
                y_val_pred.extend(outputs.view(-1).cpu().numpy())
                y_val_true.extend(y_batch.cpu().numpy())

        r2_val = r2_score(y_val_true, y_val_pred)
        mse_val = mean_squared_error(y_val_true, y_val_pred)
        avg_r2_val.append(r2_val)
        avg_mse_val.append(mse_val)

        # Early stopping
        if r2_val > best_val_r2:
            best_val_r2 = r2_val
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

        print(f"Epoch {epoch + 1}: Train R²: {r2_train:.4f}, Val R²: {r2_val:.4f}, Val MSE: {mse_val:.4f}")

    # Save the best model and results
    model.load_state_dict(best_model_state)
    avg_r2_train_final = np.mean(avg_r2_train)
    avg_mse_train_final = np.mean(avg_mse_train)
    avg_r2_val_final = np.mean(avg_r2_val)
    avg_mse_val_final = np.mean(avg_mse_val)

    print(f"\nAverage Train R²: {avg_r2_train_final:.4f}, Average Train MSE: {avg_mse_train_final:.4f}")
    print(f"Average Validation R²: {avg_r2_val_final:.4f}, Average Validation MSE: {avg_mse_val_final:.4f}")

    return model, best_val_r2

# Define hyperparameters
best_params = {
    'attention_dim': 375,
    'hidden_size': 398,
    'num_layers': 4,
    'attention_heads': 2,
    'dropout_rate': 0.041753198497107055,
    'weight_decay': 7.018481184104582e-08,
    'batch_size': 32,
    'learning_rate': 4.231457618669589e-05,
    'split_ratio': 0.9
}

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model, best_r2 = train_model(best_params, X_tensor, y_tensor, seq_len_tensor, device)

# Save the model for further analysis
torch.save(trained_model.state_dict(), "best_gru_attention_model.pth")
print(f"Best R² on validation set: {best_r2:.4f}")

# In[5]:
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, output).squeeze(1)

        # Final output
        output = self.fc(context)
        return output, attn_weights

# Hyperparameters
input_size = 1024  # Embedding dimension
hidden_size = 256
num_layers = 2
attention_dim = 128
output_size = 1
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Prepare data
embeddings = torch.tensor(longitudinal_train, dtype=torch.float32)
labels = torch.tensor(labels['labels'].values, dtype=torch.float32).view(-1, 1)
seq_lengths = torch.tensor(train_seq_len, dtype=torch.int64)

# Dataset and DataLoader
dataset = TensorDataset(embeddings, labels, seq_lengths)
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, optimizer, and loss function
model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for x_batch, y_batch, seq_len_batch in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(x_batch, seq_len_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {np.mean(train_losses):.4f}")

# Evaluation
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch, seq_len_batch in loader:
            preds, _ = model(x_batch, seq_len_batch)
            all_preds.extend(preds.squeeze(1).cpu().numpy())
            all_labels.extend(y_batch.squeeze(1).cpu().numpy())
    
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    return mse, mae, r2

# Calculate metrics for train and test sets
train_mse, train_mae, train_r2 = evaluate(model, train_loader)
test_mse, test_mae, test_r2 = evaluate(model, test_loader)

print("\nTrain Set Evaluation:")
print(f"MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")

print("\nTest Set Evaluation:")
print(f"MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# In[6]:
from sklearn.model_selection import KFold

# Define 5-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

# Store metrics for each fold
fold_metrics = {'train_mse': [], 'train_mae': [], 'train_r2': [],
                'test_mse': [], 'test_mae': [], 'test_r2': []}

# Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold + 1}/{k_folds}")
    
    # Create DataLoaders for the current fold
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    # Initialize model and optimizer
    model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop for each fold
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch, seq_len_batch in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(x_batch, seq_len_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    # Evaluate model on train and test sets for the current fold
    train_mse, train_mae, train_r2 = evaluate(model, train_loader)
    test_mse, test_mae, test_r2 = evaluate(model, test_loader)

    # Store metrics for the current fold
    fold_metrics['train_mse'].append(train_mse)
    fold_metrics['train_mae'].append(train_mae)
    fold_metrics['train_r2'].append(train_r2)
    fold_metrics['test_mse'].append(test_mse)
    fold_metrics['test_mae'].append(test_mae)
    fold_metrics['test_r2'].append(test_r2)

    # Print metrics for the current fold
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# Display average metrics across all folds
print("\nAverage Metrics Across Folds:")
for metric in fold_metrics:
    avg_metric = np.mean(fold_metrics[metric])
    print(f"{metric.capitalize()} (average): {avg_metric:.4f}")

# In[7]:
from sklearn.model_selection import KFold
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, output).squeeze(1)

    
        # Final output
        output = self.fc(context)
        return output, attn_weights

# Hyperparameters
input_size = 1024  # Embedding dimension
hidden_size = 256
num_layers = 2
attention_dim = 128
output_size = 1
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Prepare data
embeddings = torch.tensor(longitudinal_train, dtype=torch.float32)
labels = torch.tensor(labels['labels'].values, dtype=torch.float32).view(-1, 1)
seq_lengths = torch.tensor(train_seq_len, dtype=torch.int64)

# Dataset and DataLoader
dataset = TensorDataset(embeddings, labels, seq_lengths)
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Hyperparameters (set to your preferred values)
hidden_size = 256
num_layers = 2
attention_dim = 128
learning_rate = 0.001
weight_decay = 1e-4
num_epochs = 50
batch_size = 64

# Store metrics for each fold
fold_metrics = {'train_mse': [], 'train_mae': [], 'train_r2': [],
                'test_mse': [], 'test_mae': [], 'test_r2': []}

# 5-Fold Cross-Validation loop with 60-40 split
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold + 1}/{k_folds}")
    
    # Create DataLoaders for the current fold
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop for each fold
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch, seq_len_batch in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(x_batch, seq_len_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    # Evaluate model on train and test sets for the current fold
    train_mse, train_mae, train_r2 = evaluate(model, train_loader)
    test_mse, test_mae, test_r2 = evaluate(model, test_loader)

    # Store metrics for the current fold
    fold_metrics['train_mse'].append(train_mse)
    fold_metrics['train_mae'].append(train_mae)
    fold_metrics['train_r2'].append(train_r2)
    fold_metrics['test_mse'].append(test_mse)
    fold_metrics['test_mae'].append(test_mae)
    fold_metrics['test_r2'].append(test_r2)

    # Print metrics for the current fold
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# Display average metrics across all folds
print("\nAverage Metrics Across Folds:")
for metric in fold_metrics:
    avg_metric = np.mean(fold_metrics[metric])
    print(f"{metric.capitalize()} (average): {avg_metric:.4f}")

# In[8]:
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import optuna
import numpy as np

# Define the GRU model with stacked attention layers, dropout, and layer normalization
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size, dropout_rate, use_layer_norm):
        super(GRUAttentionModel, self).__init__()
        
        # Bidirectional GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size * 2)  # Double hidden_size for bidirectional

        # Stacked attention layers
        self.attention = nn.Linear(hidden_size * 2, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.attention2 = nn.Linear(hidden_size * 2, attention_dim)  # Second attention layer
        self.attention_combine2 = nn.Linear(attention_dim, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Adjusted for bidirectional GRU

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if self.use_layer_norm:
            output = self.layer_norm(output)

        # First attention mechanism
        attn_scores = torch.tanh(self.attention(output))
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, output).squeeze(1)

        # Second attention mechanism for stacked attention
        attn_scores2 = torch.tanh(self.attention2(output))
        attn_scores2 = self.attention_combine2(attn_scores2).squeeze(-1)
        attn_weights2 = F.softmax(attn_scores2, dim=1)
        attn_weights2 = attn_weights2.unsqueeze(1)
        context2 = torch.bmm(attn_weights2, output).squeeze(1)

        # Combine attention contexts and apply dropout
        final_context = self.dropout(context + context2)
        output = self.fc(final_context)
        return output, attn_weights

# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch, seq_len_batch in loader:
            preds, _ = model(x_batch, seq_len_batch)
            all_preds.extend(preds.squeeze(1).cpu().numpy())
            all_labels.extend(y_batch.squeeze(1).cpu().numpy())
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    return mse, mae, r2

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int("hidden_size", 128, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    attention_dim = trial.suggest_int("attention_dim", 64, 512, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])

    # Split the dataset into train (60%) and test (40%)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.4, shuffle=True, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        # Create DataLoaders for the current fold
        train_fold_subset = Subset(train_dataset, train_idx)
        val_fold_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_fold_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold_subset, batch_size=batch_size)

        # Initialize model and optimizer for each fold
        model = GRUAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_dim=attention_dim,
            output_size=output_size,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 5  # Can adjust based on preference

        # Training loop with early stopping
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            for x_batch, y_batch, seq_len_batch in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(x_batch, seq_len_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            # Validation loss for early stopping
            val_mse, _, _ = evaluate(model, val_loader)
            scheduler.step(val_mse)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break  # Early stopping

        # Validation metrics for current fold
        _, _, val_r2 = evaluate(model, val_loader)
        r2_scores.append(val_r2)

    # Return average R² score across folds for Optuna to optimize
    return np.mean(r2_scores)

# Remaining code for Optuna study and final evaluation as provided

# Prepare data tensors
embeddings = torch.tensor(longitudinal_train, dtype=torch.float32)
labels_tensor = torch.tensor(labels['labels'].values, dtype=torch.float32).view(-1, 1)
seq_lengths_tensor = torch.tensor(train_seq_len, dtype=torch.int64)

# Dataset
dataset = torch.utils.data.TensorDataset(embeddings, labels_tensor, seq_lengths_tensor)

# Hyperparameters
input_size = 1024  # Embedding dimension
output_size = 1
num_epochs = 100  # Increased epochs for better training

# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Best hyperparameters
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# Final evaluation on test set using best hyperparameters
best_params = study.best_params
batch_size = best_params['batch_size']

# Split the dataset into train and test sets
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.4, shuffle=True, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model with best hyperparameters
model = GRUAttentionModel(
    input_size=input_size,
    hidden_size=best_params['hidden_size'],
    num_layers=best_params['num_layers'],
    attention_dim=best_params['attention_dim'],
    output_size=output_size,
    dropout_rate=best_params['dropout_rate'],
    use_layer_norm=best_params['use_layer_norm']
)
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 10  # Adjusted for final training

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    for x_batch, y_batch, seq_len_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        outputs, _ = model(x_batch, seq_len_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    # Validation loss for early stopping
    val_mse, _, _ = evaluate(model, test_loader)
    scheduler.step(val_mse)

    if val_mse < best_val_loss:
        best_val_loss = val_mse
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Evaluate on test set
test_mse, test_mae, test_r2 = evaluate(model, test_loader)
print("\nTest Set Evaluation with Best Hyperparameters:")
print(f"MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# In[9]:
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, attention_heads, dropout_rate, output_size):
        super(GRUAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_heads = nn.ModuleList([nn.Linear(attention_dim, 1, bias=False) for _ in range(attention_heads)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Multi-head attention computation
        attention_weights = [torch.tanh(self.attention(output)) for _ in self.attention_heads]
        attention_scores = [head(attention).squeeze(-1) for attention, head in zip(attention_weights, self.attention_heads)]
        attention_scores = torch.stack(attention_scores, dim=-1)
        attention_scores = torch.softmax(attention_scores.mean(dim=-1), dim=1)

        # Weighted context vector
        context_vector = torch.sum(attention_scores.unsqueeze(-1) * output, dim=1)
        return self.fc(self.dropout(context_vector)), attention_scores

# Define a base parameter set from the best trial to vary each parameter
best_params = {
    'attention_dim': 387,
    'hidden_size': 391,
    'num_layers': 4,
    'attention_heads': 2,
    'dropout_rate': 0.04533684526428862,
    'weight_decay': 1.7046661134499106e-08,
    'batch_size': 32,
    'learning_rate': 3.435180767024428e-05,
    'split_ratio': 0.9
}

# Objective function with variations around the best parameters
def objective(trial):
    # Vary one parameter while keeping others constant
    attention_dim = trial.suggest_int("attention_dim", best_params['attention_dim'] - 20, best_params['attention_dim'] + 20)
    hidden_size = trial.suggest_int("hidden_size", best_params['hidden_size'] - 20, best_params['hidden_size'] + 20)
    num_layers = trial.suggest_int("num_layers", max(1, best_params['num_layers'] - 1), best_params['num_layers'] + 1)
    attention_heads = trial.suggest_int("attention_heads", max(1, best_params['attention_heads'] - 1), best_params['attention_heads'] + 1)
    dropout_rate = trial.suggest_float("dropout_rate", max(0.0, best_params['dropout_rate'] - 0.02), min(0.5, best_params['dropout_rate'] + 0.02))
    weight_decay = trial.suggest_loguniform("weight_decay", best_params['weight_decay'] / 10, best_params['weight_decay'] * 10)
    batch_size = trial.suggest_categorical("batch_size", [best_params['batch_size'], best_params['batch_size'] * 2])
    learning_rate = trial.suggest_loguniform("learning_rate", best_params['learning_rate'] / 2, best_params['learning_rate'] * 2)
    split_ratio = trial.suggest_categorical("split_ratio", [0.6, 0.7, 0.8, 0.9])

    # Data split and loaders
    split_point = int(len(X_tensor) * split_ratio)
    X_train, X_val = X_tensor[:split_point], X_tensor[split_point:]
    y_train, y_val = y_tensor[:split_point], y_tensor[split_point:]
    seq_len_train, seq_len_val = seq_len_tensor[:split_point], seq_len_tensor[split_point:]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model and optimizer setup
    model = GRUAttentionModel(
        input_size=X_tensor.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        output_size=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Training loop
    best_val_r2 = -np.inf
    patience_counter = 0
    num_epochs = 100
    early_stopping_patience = 15

    for epoch in range(num_epochs):
        model.train()
        y_train_pred, y_train_true = [], []
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            y_train_pred.extend(outputs.view(-1).detach().cpu().numpy())
            y_train_true.extend(y_batch.detach().cpu().numpy())

        r2_train = r2_score(y_train_true, y_train_pred)

        # Validation
        model.eval()
        y_val_pred, y_val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model(X_batch, seq_len_batch)
                y_val_pred.extend(outputs.view(-1).cpu().numpy())
                y_val_true.extend(y_batch.cpu().numpy())

        r2_val = r2_score(y_val_true, y_val_pred)
        mse_val = mean_squared_error(y_val_true, y_val_pred)
        scheduler.step(mse_val)

        trial.report(r2_val, epoch)
        if r2_val > best_val_r2:
            best_val_r2 = r2_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

        print(f"Trial {trial.number}, Epoch {epoch}: Train R²: {r2_train:.4f}, Val R²: {r2_val:.4f}, Val MSE: {mse_val:.4f}")

    return best_val_r2

# Run Optuna study with expanded trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)  # Expanding trials for more comprehensive exploration

# Output best trial results
print("Best trial:")
print(study.best_trial)

# In[10]:
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error

# Your model setup, data, and preprocessing code goes here

# Objective function for Optuna
def objective(trial):
    # Hyperparameters with a focus on ranges near top-performing trials
    attention_dim = trial.suggest_int("attention_dim", 240, 260)
    hidden_size = trial.suggest_int("hidden_size", 270, 310)
    num_layers = trial.suggest_int("num_layers", 2, 3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 2e-5, 0.0001)
    split_ratio = trial.suggest_categorical("split_ratio", [0.8, 0.9])
    
    # Data split with increased test set size
    train_size = int(len(X_tensor) * split_ratio)
    val_size = len(X_tensor) - train_size
    train_data, val_data = random_split(
        TensorDataset(X_tensor, y_tensor, seq_len_tensor), [train_size, val_size]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer, and scheduler setup
    model = GRUAttentionModel(
        input_size=X_tensor.shape[2], 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        attention_dim=attention_dim, 
        output_size=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Training loop
    num_epochs = 20  # Adjust as needed for more training rounds
    for epoch in range(num_epochs):
        model.train()
        y_train_pred, y_train_true = [], []
        
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = (
                X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            )
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            y_train_pred.extend(outputs.detach().squeeze().cpu().numpy())
            y_train_true.extend(y_batch.cpu().numpy())

        # Validation loop
        model.eval()
        y_val_pred, y_val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = (
                    X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                )
                outputs, _ = model(X_batch, seq_len_batch)
                y_val_pred.extend(outputs.squeeze().cpu().numpy())
                y_val_true.extend(y_batch.cpu().numpy())

        # R² calculation for train and validation
        r2_train = r2_score(y_train_true, y_train_pred)
        r2_val = r2_score(y_val_true, y_val_pred)

        # Adjust learning rate based on validation R² improvement
        scheduler.step(r2_val)

        # Early stopping criteria
        trial.report(r2_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return r2_val

# Run the Optuna study with refined parameters and additional trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Output results
print("Best trial:")
print(study.best_trial)
print("Top trials:")
for trial in sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]:
    print(f"Trial {trial.number}: R² = {trial.value:.4f}, Params = {trial.params}")
