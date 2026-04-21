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
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check for GPU availability
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

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)  # Attention mechanism
        self.attention_combine = nn.Linear(attention_dim, 1)    # Combine attention scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, output).squeeze(1)  # (batch_size, hidden_size)

        output = self.fc(context)  # (batch_size, output_size)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128  # Adjust the attention dimension if needed
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics and attention weights
all_train_losses, all_val_losses = [], []
all_attention_weights = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    seq_len_train, seq_len_val = seq_len[train_index], seq_len[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, 1).to(device)
    criterion = nn.L1Loss()  # Using MAE as the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch, seq_len_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        epoch_val_losses = []
        fold_attention_weights = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, attention_weights = model(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_losses.append(loss.item())
                
                # Collect attention weights for the batch
                fold_attention_weights.extend(attention_weights.cpu().numpy())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))
    
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_attention_weights.append(fold_attention_weights)

# Average losses over folds for each epoch
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

def calculate_metrics(y_true, y_pred, num_features):
    """
    Calculate and return MAE, MSE, R², Adjusted R², and correlation coefficient r.
    """
    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R²
    n = len(y_true)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))

    # Correlation coefficient
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r = correlation_xy

    return mae, mse, r2, adj_r2, r

# After cross-validation and training, evaluate on the entire test set

# Final evaluation on train set
model.eval()
with torch.no_grad():
    y_train_pred, _ = model(X_train_tensor.to(device), train_seq_len_tensor.to(device))
    y_train_pred = y_train_pred.squeeze().cpu().numpy()
    y_train_true = y_train_tensor.cpu().numpy()

    num_features = X_train_tensor.shape[2]  # Number of features
    mae_train, mse_train, r2_train, adj_r2_train, r_train = calculate_metrics(y_train_true, y_train_pred, num_features)

# Final evaluation on test set
with torch.no_grad():
    y_test_pred, _ = model(X_test_tensor.to(device), test_seq_len_tensor.to(device))
    y_test_pred = y_test_pred.squeeze().cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    mae_test, mse_test, r2_test, adj_r2_test, r_test = calculate_metrics(y_test_true, y_test_pred, num_features)

# Print final metrics for train and test sets
print(f"Train Metrics:")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MSE (Train): {mse_train:.4f}")
print(f"R² (Train): {r2_train:.4f}")
print(f"Adjusted R² (Train): {adj_r2_train:.4f}")
print(f"Correlation coefficient r (Train): {r_train:.4f}")

print("\nTest Metrics:")
print(f"MAE (Test): {mae_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"Adjusted R² (Test): {adj_r2_test:.4f}")
print(f"Correlation coefficient r (Test): {r_test:.4f}")

# Save the trained model
torch.save(model.state_dict(), '2gru_attention_model.pth')

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_train_losses, label='Train Loss')
plt.plot(range(num_epochs), avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Visualize Attention Weights for a Specific Transcript
def plot_attention_weights(attention_weights, transcript):
    attention_weights = np.array(attention_weights).squeeze()  # Ensure attention_weights is a 1D array
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(attention_weights)), attention_weights, align='center')
    plt.xticks(range(len(attention_weights)), transcript, rotation=90)
    plt.title('Attention Weights Across Transcript')
    plt.xlabel('Transcript Part')
    plt.ylabel('Attention Weight')
    plt.show()

# In[2]:
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the test.csv file containing test transcripts
test_data = pd.read_csv('Otest.csv')

# Filter out transcripts based on missing_test_ids
missing_test_ids = {4234, 4274, 4302, 4412, 4415, 4420, 4501}
valid_transcripts = test_data[~test_data['PARTID'].isin(missing_test_ids)].copy()

# Map the original indices of the valid transcripts for later use
valid_indices = valid_transcripts.index.tolist()

# Print debug information
print(f"Total transcripts in 'test_data': {len(test_data)}")
print(f"Total valid transcripts after filtering: {len(valid_transcripts)}")
print(f"Total valid indices: {len(valid_indices)}")
print(f"Total longitudinal_test samples: {len(longitudinal_test)}")

# Filter the valid_indices to ensure all are within the range of longitudinal_test
valid_indices = [i for i in valid_indices if i < len(longitudinal_test)]

# Print the updated number of valid indices
print(f"Filtered valid indices to match 'longitudinal_test': {len(valid_indices)}")

# Filter the embeddings (longitudinal_test) to match valid transcripts
X_test_tensor = torch.tensor([longitudinal_test[i] for i in valid_indices], dtype=torch.float32)
test_seq_len_tensor = torch.tensor(
    [torch.count_nonzero(torch.any(seq != 0, dim=1)) for seq in X_test_tensor], dtype=torch.int64
)

# Print debug information for the embeddings after filtering
print(f"Total test samples in 'X_test_tensor' after filtering: {len(X_test_tensor)}")

# Create a DataLoader for the valid transcripts
dataset = TensorDataset(X_test_tensor, test_seq_len_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Print the length of the DataLoader
print(f"Total batches in DataLoader after filtering: {len(loader)}")

# Store the predicted values for each transcript
predicted_values = []
attention_weights_all = []

# Function to get predictions and attention weights
with torch.no_grad():
    for i, (X_batch, seq_len_batch) in enumerate(loader):
        X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)  # Move to device
        output, attention_weights = model(X_batch, seq_len_batch)
        original_idx = valid_indices[i]  # Map the DataLoader index to the original valid index
        predicted_values.append((output.item(), original_idx, attention_weights))
        attention_weights_all.append(attention_weights)

# Sort the predicted values in descending order to get the highest ranked transcripts
predicted_values.sort(reverse=True, key=lambda x: x[0])

# Select the top 5 highest ranked transcripts
top_5 = predicted_values[:5]

# Function to extract the highest-ranked window and its words
def extract_highest_attention_window_words(transcript_text, attention_weights, seq_len):
    attention_weights = attention_weights.squeeze()[:seq_len]  # Extract valid attention weights
    words = transcript_text.split()
    num_words_per_window = len(words) // len(attention_weights)
    max_idx = torch.argmax(attention_weights).item()
    start_idx = max_idx * num_words_per_window
    end_idx = min(start_idx + num_words_per_window, len(words))
    highest_attention_words = " ".join(words[start_idx:end_idx])
    return highest_attention_words

# Plot attention heatmaps for the top 5 transcripts with the highest attention window
for predicted_value, original_idx, attention_weights in top_5:
    transcript_text = valid_transcripts.iloc[original_idx]['text']
    transcript_id = valid_transcripts.iloc[original_idx]['PARTID']
    seq_len = test_seq_len_tensor[valid_indices.index(original_idx)].item()
    highest_attention_words = extract_highest_attention_window_words(transcript_text, attention_weights[0], seq_len)
    
    # Print the highest-ranked window words
    print(f"Transcript ID: {transcript_id}")
    print(f"Highest-ranked window words: {highest_attention_words}\n")

    # Plot the heatmap for visual inspection
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.squeeze()[:seq_len].cpu().unsqueeze(0).numpy(), cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)  # Add dimension to make it 2D
    plt.title(f'Attention Weights Heatmap for Transcript ID: {transcript_id}')
    plt.show()

# In[3]:
import pickle

# Set the target transcript PARTID
target_transcript_id = 4247

# Check that the transcript exists in your valid_transcripts DataFrame
if not (valid_transcripts['PARTID'] == target_transcript_id).any():
    print(f"Transcript with PARTID {target_transcript_id} was not found in valid_transcripts.")
else:
    # Get the row corresponding to the target transcript
    target_df = valid_transcripts[valid_transcripts['PARTID'] == target_transcript_id]
    target_df_index = target_df.index[0]  # This is the DataFrame's index for the transcript

    # Find the position in valid_indices that matches this DataFrame index
    if target_df_index not in valid_indices:
        print(f"Transcript index {target_df_index} not found in valid_indices.")
    else:
        position = valid_indices.index(target_df_index)
        print(f"Transcript {target_transcript_id} found at position {position} in X_test_tensor.")

        # Extract the corresponding test embedding and sequence length
        target_embedding = X_test_tensor[position].unsqueeze(0)  # Add batch dimension
        target_seq_length = test_seq_len_tensor[position].unsqueeze(0)

        # Switch the model to evaluation mode and compute attention weights
        model.eval()
        with torch.no_grad():
            _, attention_weights = model(target_embedding.to(device), target_seq_length.to(device))
        
        # Convert the attention weights to a NumPy array
        attention_weights_np = attention_weights.cpu().numpy()

        # Save the attention weights to a file
        output_filename = 'transcript_4247_attention_weights.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(attention_weights_np, f)

        print(f"Attention weights for transcript {target_transcript_id} have been saved to '{output_filename}'.")

# In[4]:
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch.nn.functional as F

# Load necessary files
test_data = pd.read_csv('Otest.csv')
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)

# Filter valid PARTIDs from Ctest.csv
missing_test_ids = {4234, 4274, 4302, 4367, 4412, 4415, 4420, 4501}
valid_transcripts = test_data[~test_data['PARTID'].isin(missing_test_ids)].copy()
valid_indices = valid_transcripts.index.tolist()

# Ensure indices are within range
valid_indices = [i for i in valid_indices if i < len(longitudinal_test)]
X_test_tensor = torch.tensor([longitudinal_test[i] for i in valid_indices], dtype=torch.float32)
test_seq_len_tensor = torch.tensor(
    [torch.count_nonzero(torch.any(seq != 0, dim=1)) for seq in X_test_tensor], dtype=torch.int64
)

# Create DataLoader
dataset = TensorDataset(X_test_tensor, test_seq_len_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Store predictions and attention weights
predicted_scores = []
attention_windows = []
PARTIDs = []
ground_truth_scores = []

# Modify extract_top_attention_windows to handle cases where there are fewer than 5 attention scores
def extract_top_attention_windows(transcript_text, attention_weights, seq_len, num_windows=5):
    attention_weights = attention_weights.squeeze()[:seq_len]  # Only valid weights up to sequence length
    words = transcript_text.split()
    num_words_per_window = max(len(words) // len(attention_weights), 1)  # Ensure no division by zero
    
    # Adjust num_windows to avoid selecting more indices than available
    num_windows = min(num_windows, len(attention_weights))
    top_indices = torch.topk(attention_weights, num_windows).indices.cpu().numpy()
    
    top_windows = []
    for idx in top_indices:
        start_idx = idx * num_words_per_window
        end_idx = min(start_idx + num_words_per_window, len(words))
        top_windows.append(" ".join(words[start_idx:end_idx]))
    
    # Add empty strings if fewer than 5 windows are available
    while len(top_windows) < 5:
        top_windows.append("")

    return top_windows

# Collect data for each transcript
model.eval()
with torch.no_grad():
    for i, (X_batch, seq_len_batch) in enumerate(loader):
        X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
        output, attention_weights = model(X_batch, seq_len_batch)
        
        # Get corresponding transcript details using iloc[i]
        transcript_text = valid_transcripts.iloc[i]['text']
        transcript_id = valid_transcripts.iloc[i]['PARTID']
        ground_truth_score = valid_transcripts.iloc[i]['labels']
        predicted_score = output.item()
        
        # Extract top 5 attention windows
        top_windows = extract_top_attention_windows(transcript_text, attention_weights[0], seq_len_batch.item())
        
        # Append data to lists
        PARTIDs.append(transcript_id)
        ground_truth_scores.append(ground_truth_score)
        predicted_scores.append(predicted_score)
        attention_windows.append(top_windows)


# Convert to DataFrame
data = {
    'PARTID': PARTIDs,
    'Top_Window_1': [win[0] if len(win) > 0 else "" for win in attention_windows],
    'Top_Window_2': [win[1] if len(win) > 1 else "" for win in attention_windows],
    'Top_Window_3': [win[2] if len(win) > 2 else "" for win in attention_windows],
    'Top_Window_4': [win[3] if len(win) > 3 else "" for win in attention_windows],
    'Top_Window_5': [win[4] if len(win) > 4 else "" for win in attention_windows],
    'Ground_Truth_Score': ground_truth_scores,
    'Predicted_Score': predicted_scores
}

df = pd.DataFrame(data)
df.to_csv('O_Top_Attention_Windows.csv', index=False)

print("CSV file 'O_Top_Attention_Windows.csv' created successfully.")

# In[5]:
# Reload test data
test_data = pd.read_csv('Otest.csv')  # Ensure this contains PARTID and other relevant data
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)
with open('test_seq_len', 'rb') as handle:
    test_seq_len = pickle.load(handle)

# Filter out the missing_test_ids
missing_test_ids = {4234, 4274, 4302, 4367, 4412, 4415, 4420, 4501}
valid_transcripts = test_data[~test_data['PARTID'].isin(missing_test_ids)].copy()

# Map valid indices to ensure they align with the embeddings and labels
valid_indices = valid_transcripts.index.tolist()
valid_indices = [i for i in valid_indices if i < len(longitudinal_test)]  # Ensure indices are in range

# Filter the embeddings, labels, and sequence lengths
X_test_tensor = torch.tensor([longitudinal_test[i] for i in valid_indices], dtype=torch.float32)
y_test_tensor = torch.tensor([test_labels[i] for i in valid_indices], dtype=torch.float32)
seq_len_test_tensor = torch.tensor(
    [torch.count_nonzero(torch.any(torch.tensor(longitudinal_test[i]) != 0, dim=1)) for i in valid_indices],
    dtype=torch.int64
)

# Debugging Information
print(f"Filtered transcripts: {len(valid_transcripts)}")
print(f"Valid indices: {len(valid_indices)}")
print(f"Filtered embeddings shape: {X_test_tensor.shape}")

# In[6]:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle

# Reload test data
test_data = pd.read_csv('Otest.csv')  # Ensure this contains PARTID and other relevant data
with open('longitudinal_test', 'rb') as handle:
    longitudinal_test = pickle.load(handle)
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)
with open('test_seq_len', 'rb') as handle:
    test_seq_len = pickle.load(handle)

# Filter out the missing_test_ids
missing_test_ids = {4234, 4274, 4302, 4367, 4412, 4415, 4420, 4501}
valid_transcripts = test_data[~test_data['PARTID'].isin(missing_test_ids)].copy()

# Map valid indices to ensure they align with the embeddings and labels
valid_indices = valid_transcripts.index.tolist()
valid_indices = [i for i in valid_indices if i < len(longitudinal_test)]  # Ensure indices are in range

# Filter the embeddings, labels, and sequence lengths
X_test_tensor = torch.tensor([longitudinal_test[i] for i in valid_indices], dtype=torch.float32)
y_test_tensor = torch.tensor([test_labels[i] for i in valid_indices], dtype=torch.float32)
seq_len_test_tensor = torch.tensor(
    [torch.count_nonzero(torch.any(torch.tensor(longitudinal_test[i]) != 0, dim=1)) for i in valid_indices],
    dtype=torch.int64
)

# Debugging Information
print(f"Filtered transcripts: {len(valid_transcripts)}")
print(f"Valid indices: {len(valid_indices)}")
print(f"Filtered embeddings shape: {X_test_tensor.shape}")

# Define model architecture
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, output).squeeze(1)

        output = self.fc(context)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128
output_size = 1
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAttentionModel(X_test_tensor.shape[2], hidden_size, num_layers, attention_dim, output_size).to(device)
model.load_state_dict(torch.load('2gru_attention_model.pth'))
model.eval()

# Mask the most significant attention window
def mask_highest_attention_window(X_batch, attention_weights, seq_len_batch):
    masked_batch = X_batch.clone()
    for i in range(len(X_batch)):
        valid_seq_len = seq_len_batch[i].item()
        attn_weights = attention_weights[i, :valid_seq_len]
        max_idx = torch.argmax(attn_weights).item()

        # Zero out the embedding corresponding to the highest attention window
        masked_batch[i, max_idx, :] = 0.0
    return masked_batch

masked_X_test = torch.clone(X_test_tensor)
with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        X_batch = X_test_tensor[i:i+batch_size].to(device)
        seq_len_batch = seq_len_test_tensor[i:i+batch_size].to(device)
        _, attention_weights = model(X_batch, seq_len_batch)
        masked_X_test[i:i+batch_size] = mask_highest_attention_window(X_batch, attention_weights, seq_len_batch)

# Evaluate the model on masked test data
retrained_predictions = []
with torch.no_grad():
    for i in range(0, len(masked_X_test), batch_size):
        X_batch = masked_X_test[i:i+batch_size].to(device)
        seq_len_batch = seq_len_test_tensor[i:i+batch_size].to(device)
        outputs, _ = model(X_batch, seq_len_batch)
        retrained_predictions.extend(outputs.cpu().numpy())

# Load the CSV and append retrained predictions
df = pd.read_csv('O_Top_Attention_Windows.csv')
if len(retrained_predictions) == len(df):
    df['Retrained_Predicted_Score'] = retrained_predictions
    df['Score_Impact'] = df['Predicted_Score'] - df['Retrained_Predicted_Score']
    df.to_csv('O_Top_Attention_Windows_Updated.csv', index=False)
    print("Retrained predictions and score impact saved in 'O_Top_Attention_Windows_Updated.csv'.")
else:
    print(f"Error: Mismatch between retrained predictions ({len(retrained_predictions)}) and test transcript indices ({len(df)}).")

# In[7]:
# Print the length of longitudinal_train
print(f"Length of longitudinal_train: {len(longitudinal_train)}")

# Print the shape of each sequence in longitudinal_test
print(f"Shape of each sequence in longitudinal_train: {[sample.shape for sample in longitudinal_train]}")

# Print the first element of longitudinal_train
print("First element of longitudinal_train:", longitudinal_train[0])

# In[8]:
print(f"Length of longitudinal_train: {len(longitudinal_test)}")
print(f"Shape of each sequence in longitudinal_train: {[sample.shape for sample in longitudinal_train]}")

# In[9]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check for GPU availability
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

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)  # Attention mechanism
        self.attention_combine = nn.Linear(attention_dim, 1)    # Combine attention scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, output).squeeze(1)  # (batch_size, hidden_size)

        output = self.fc(context)  # (batch_size, output_size)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128  # Adjust the attention dimension if needed
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics
all_train_losses, all_val_losses = [], []
all_mae_train, all_mae_val = [], []
all_mse_train, all_mse_val = [], []
all_r2_train, all_r2_val = [], []
all_r_train, all_r_val = [], []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    seq_len_train, seq_len_val = seq_len[train_index], seq_len[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, 1).to(device)
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
            outputs, _ = model(X_batch, seq_len_batch)  # Unpack the tuple
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model(X_batch, seq_len_batch)  # Unpack the tuple
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))

        # Unpack the tuple for predictions
        y_train_pred, _ = model(X_train.to(device), seq_len_train.to(device))
        y_train_pred = y_train_pred.squeeze()

        y_val_pred, _ = model(X_val.to(device), seq_len_val.to(device))
        y_val_pred = y_val_pred.squeeze()

        y_train_np = y_train.cpu().numpy()
        y_train_pred_np = y_train_pred.detach().cpu().numpy()
        mae_train = mean_absolute_error(y_train_np, y_train_pred_np)
        mse_train = mean_squared_error(y_train_np, y_train_pred_np)
        r2_train = r2_score(y_train_np, y_train_pred_np)
        correlation_matrix_train = np.corrcoef(y_train_np, y_train_pred_np)
        correlation_xy_train = correlation_matrix_train[0, 1]
        r_train = correlation_xy_train

        y_val_np = y_val.cpu().numpy()
        y_val_pred_np = y_val_pred.detach().cpu().numpy()
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
torch.save(model.state_dict(), '1gru_attention_model.pth')

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

# In[10]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import optuna

# Check for GPU availability
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
input_size = X_train_tensor.shape[2]

# Define the GRU model with attention and dropout
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size, dropout_rate=0.5):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, output).squeeze(1)
        context = self.dropout(context)
        output = self.fc(context)
        return output, attn_weights

# Hyperparameters for tuning
batch_size = 64
num_epochs = 50

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    attention_dim = trial.suggest_int('attention_dim', 32, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_r2_scores = []

    for train_index, val_index in kf.split(X_train_tensor):
        X_train_cv, X_val_cv = X_train_tensor[train_index], X_train_tensor[val_index]
        y_train_cv, y_val_cv = y_train_tensor[train_index], y_train_tensor[val_index]
        seq_len_train_cv, seq_len_val_cv = train_seq_len_tensor[train_index], train_seq_len_tensor[val_index]

        train_dataset_cv = TensorDataset(X_train_cv, y_train_cv, seq_len_train_cv)
        val_dataset_cv = TensorDataset(X_val_cv, y_val_cv, seq_len_val_cv)

        train_loader_cv = DataLoader(train_dataset_cv, batch_size=batch_size, shuffle=True)
        val_loader_cv = DataLoader(val_dataset_cv, batch_size=batch_size, shuffle=False)

        # Initialize model
        model_cv = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, 1, dropout_rate=dropout_rate).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model_cv.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training loop
        for epoch in range(num_epochs):
            model_cv.train()
            for X_batch, y_batch, seq_len_batch in train_loader_cv:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model_cv(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        # Validation
        model_cv.eval()
        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader_cv:
                X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model_cv(X_batch, seq_len_batch)
                y_val_true.extend(y_batch.numpy())
                y_val_pred.extend(outputs.squeeze().cpu().numpy())

        # Calculate R² score
        r2 = r2_score(y_val_true, y_val_pred)
        val_r2_scores.append(r2)

        # Report intermediate value to Optuna
        trial.report(r2, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the average R² score over the folds
    return np.mean(val_r2_scores)

# Create a study and optimize it
study = optuna.create_study(direction='maximize')  # Set direction to 'maximize' for R²
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial

print(f"  Value (R²): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Use the best hyperparameters
hidden_size = trial.params['hidden_size']
num_layers = trial.params['num_layers']
attention_dim = trial.params['attention_dim']
learning_rate = trial.params['learning_rate']
weight_decay = trial.params['weight_decay']
dropout_rate = trial.params['dropout_rate']

# Retrain the model on the full training set
full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_seq_len_tensor)
full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

# Initialize a new model instance with the best hyperparameters
model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, 1, dropout_rate=dropout_rate).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop on full training data
for epoch in tqdm(range(num_epochs), desc="Training on Full Data"):
    model.train()
    for X_batch, y_batch, seq_len_batch in full_train_loader:
        X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
        optimizer.zero_grad()
        outputs, _ = model(X_batch, seq_len_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

print("Finished retraining on the full training set.")

# Evaluate on the test set
model.eval()
with torch.no_grad():
    y_test_pred, _ = model(X_test_tensor.to(device), test_seq_len_tensor.to(device))
    y_test_pred = y_test_pred.squeeze().cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    # Calculate metrics
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    mse_test = mean_squared_error(y_test_true, y_test_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    correlation_matrix = np.corrcoef(y_test_true, y_test_pred)
    correlation_xy = correlation_matrix[0,1]
    r_test = correlation_xy

print("\nTest Metrics:")
print(f"MAE (Test): {mae_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"Correlation coefficient r (Test): {r_test:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'final_gru_attention_model.pth')

# In[11]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check for GPU availability
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


# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)  # Attention mechanism
        self.attention_combine = nn.Linear(attention_dim, 1)    # Combine attention scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, output).squeeze(1)  # (batch_size, hidden_size)

        output = self.fc(context)  # (batch_size, output_size)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128  # Adjust the attention dimension if needed
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics and attention weights
all_train_losses, all_val_losses = [], []
all_attention_weights = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    seq_len_train, seq_len_val = seq_len[train_index], seq_len[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUAttentionModel(input_size, hidden_size, num_layers, attention_dim, 1).to(device)
    criterion = nn.L1Loss()  # Using MAE as the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch, seq_len_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        epoch_val_losses = []
        fold_attention_weights = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, attention_weights = model(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_losses.append(loss.item())
                
                # Collect attention weights for the batch
                fold_attention_weights.extend(attention_weights.cpu().numpy())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))
    
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_attention_weights.append(fold_attention_weights)

# Average losses over folds for each epoch
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

def calculate_metrics(y_true, y_pred, num_features):
    """
    Calculate and return MAE, MSE, R², Adjusted R², and correlation coefficient r.
    """
    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R²
    n = len(y_true)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))

    # Correlation coefficient
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r = correlation_xy

    return mae, mse, r2, adj_r2, r

# After cross-validation and training, evaluate on the entire test set

# Final evaluation on train set
model.eval()
with torch.no_grad():
    y_train_pred, _ = model(X_train_tensor.to(device), train_seq_len_tensor.to(device))
    y_train_pred = y_train_pred.squeeze().cpu().numpy()
    y_train_true = y_train_tensor.cpu().numpy()

    num_features = X_train_tensor.shape[2]  # Number of features
    mae_train, mse_train, r2_train, adj_r2_train, r_train = calculate_metrics(y_train_true, y_train_pred, num_features)

# Final evaluation on test set
with torch.no_grad():
    y_test_pred, _ = model(X_test_tensor.to(device), test_seq_len_tensor.to(device))
    y_test_pred = y_test_pred.squeeze().cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    mae_test, mse_test, r2_test, adj_r2_test, r_test = calculate_metrics(y_test_true, y_test_pred, num_features)

# Print final metrics for train and test sets
print(f"Train Metrics:")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MSE (Train): {mse_train:.4f}")
print(f"R² (Train): {r2_train:.4f}")
print(f"Adjusted R² (Train): {adj_r2_train:.4f}")
print(f"Correlation coefficient r (Train): {r_train:.4f}")

print("\nTest Metrics:")
print(f"MAE (Test): {mae_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"Adjusted R² (Test): {adj_r2_test:.4f}")
print(f"Correlation coefficient r (Test): {r_test:.4f}")

# Save the trained model
torch.save(model.state_dict(), '2gru_attention_model.pth')

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_train_losses, label='Train Loss')
plt.plot(range(num_epochs), avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Visualize Attention Weights for a Specific Transcript
def plot_attention_weights(attention_weights, transcript):
    attention_weights = np.array(attention_weights).squeeze()  # Ensure attention_weights is a 1D array
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(attention_weights)), attention_weights, align='center')
    plt.xticks(range(len(attention_weights)), transcript, rotation=90)
    plt.title('Attention Weights Across Transcript')
    plt.xlabel('Transcript Part')
    plt.ylabel('Attention Weight')
    plt.show()

# In[12]:
import pickle
import numpy as np

# Load embeddings, labels, and sequence lengths
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

# Display sample data from each file
num_samples = 3  # Adjust this to show more or fewer samples

print("Samples from longitudinal_train embeddings:")
for i, embedding in enumerate(longitudinal_train[:num_samples]):
    print(f"Sample {i+1} Embedding:\n{np.array(embedding)}\n")

print("\nSamples from longitudinal_test embeddings:")
for i, embedding in enumerate(longitudinal_test[:num_samples]):
    print(f"Sample {i+1} Embedding:\n{np.array(embedding)}\n")

print("\nSamples from train_labels:")
print(train_labels[:num_samples])

print("\nSamples from test_labels:")
print(test_labels[:num_samples])

print("\nSamples from train_seq_len (padding information):")
print(train_seq_len[:num_samples])

print("\nSamples from test_seq_len (padding information):")
print(test_seq_len[:num_samples])

# In[13]:
#kfold random split
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Check for GPU availability
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

# Concatenate longitudinal_train and longitudinal_test along with their labels
longitudinal_combined = np.concatenate((longitudinal_train, longitudinal_test), axis=0)
labels_combined = np.concatenate((train_labels, test_labels), axis=0)
seq_len_combined = np.concatenate((train_seq_len, test_seq_len), axis=0)

# Convert to PyTorch tensors
X_tensor = torch.tensor(longitudinal_combined, dtype=torch.float32)
y_tensor = torch.tensor(labels_combined, dtype=torch.float32)
seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_combined], dtype=torch.int64)

# Define 60-40 train-test split
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size

# Split data
train_data, test_data = random_split(
    TensorDataset(X_tensor, y_tensor, seq_len_tensor),
    [train_size, test_size],
    generator=torch.manual_seed(42)
)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)  # Attention mechanism
        self.attention_combine = nn.Linear(attention_dim, 1)    # Combine attention scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, output).squeeze(1)  # (batch_size, hidden_size)

        output = self.fc(context)  # (batch_size, output_size)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Initialize metric tracking
all_train_losses, all_val_losses = [], []

for train_index, val_index in kf.split(X_tensor):
    X_train, X_val = X_tensor[train_index], X_tensor[val_index]
    y_train, y_val = y_tensor[train_index], y_tensor[val_index]
    seq_len_train, seq_len_val = seq_len_tensor[train_index], seq_len_tensor[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUAttentionModel(input_size=X_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers, attention_dim=attention_dim, output_size=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))

        # Validation phase
        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss_epoch.append(loss.item())

        val_losses.append(np.mean(val_loss_epoch))

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

# Average losses over folds for each epoch
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

# Final evaluation on train set
model.eval()
with torch.no_grad():
    y_train_pred, _ = model(X_tensor[train_index].to(device), seq_len_tensor[train_index].to(device))
    y_train_pred = y_train_pred.squeeze().cpu().numpy()
    y_train_true = y_tensor[train_index].cpu().numpy()
    mae_train, mse_train, r2_train = mean_absolute_error(y_train_true, y_train_pred), mean_squared_error(y_train_true, y_train_pred), r2_score(y_train_true, y_train_pred)

# Final evaluation on test set
with torch.no_grad():
    y_test_pred, _ = model(X_tensor[val_index].to(device), seq_len_tensor[val_index].to(device))
    y_test_pred = y_test_pred.squeeze().cpu().numpy()
    y_test_true = y_tensor[val_index].cpu().numpy()
    mae_test, mse_test, r2_test = mean_absolute_error(y_test_true, y_test_pred), mean_squared_error(y_test_true, y_test_pred), r2_score(y_test_true, y_test_pred)

# Print final metrics for train and test sets
print(f"Train Metrics:")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MSE (Train): {mse_train:.4f}")
print(f"R² (Train): {r2_train:.4f}")

print("\nTest Metrics:")
print(f"MAE (Test): {mae_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Test): {r2_test:.4f}")

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_train_losses, label='Train Loss')
plt.plot(range(num_epochs), avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# In[14]:
#kfold seq split
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Check for GPU availability
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

# Concatenate longitudinal_train and longitudinal_test along with their labels
longitudinal_combined = np.concatenate((longitudinal_train, longitudinal_test), axis=0)
labels_combined = np.concatenate((train_labels, test_labels), axis=0)
seq_len_combined = np.concatenate((train_seq_len, test_seq_len), axis=0)

# Convert to PyTorch tensors
X_tensor = torch.tensor(longitudinal_combined, dtype=torch.float32)
y_tensor = torch.tensor(labels_combined, dtype=torch.float32)
seq_len_tensor = torch.tensor([np.count_nonzero(np.any(seq != 0, axis=1)) for seq in longitudinal_combined], dtype=torch.int64)

# Define 60-40 train-test split
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size

# Split data
train_data, test_data = random_split(
    TensorDataset(X_tensor, y_tensor, seq_len_tensor),
    [train_size, test_size],
    generator=torch.manual_seed(42)
)


# Cross-validation setup for sequential split
kf = KFold(n_splits=5, shuffle=False)


# Define the GRU model with attention
class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, output_size):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, attention_dim)  # Attention mechanism
        self.attention_combine = nn.Linear(attention_dim, 1)    # Combine attention scores
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Compute attention scores
        attn_scores = self.attention(output)
        attn_scores = self.attention_combine(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute weighted sum of outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, output).squeeze(1)  # (batch_size, hidden_size)

        output = self.fc(context)  # (batch_size, output_size)
        return output, attn_weights

# Hyperparameters
hidden_size = 256
num_layers = 2
attention_dim = 128
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4
batch_size = 64

# Initialize metric tracking
all_train_losses, all_val_losses = [], []

for train_index, val_index in kf.split(X_tensor):
    X_train, X_val = X_tensor[train_index], X_tensor[val_index]
    y_train, y_val = y_tensor[train_index], y_tensor[val_index]
    seq_len_train, seq_len_val = seq_len_tensor[train_index], seq_len_tensor[val_index]

    train_dataset = TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = TensorDataset(X_val, y_val, seq_len_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUAttentionModel(input_size=X_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers, attention_dim=attention_dim, output_size=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))

        # Validation phase
        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for X_batch, y_batch, seq_len_batch in val_loader:
                X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
                outputs, _ = model(X_batch, seq_len_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss_epoch.append(loss.item())

        val_losses.append(np.mean(val_loss_epoch))

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

# Average losses over folds for each epoch
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

# Final evaluation on train set
model.eval()
with torch.no_grad():
    y_train_pred, _ = model(X_tensor[train_index].to(device), seq_len_tensor[train_index].to(device))
    y_train_pred = y_train_pred.squeeze().cpu().numpy()
    y_train_true = y_tensor[train_index].cpu().numpy()
    mae_train, mse_train, r2_train = mean_absolute_error(y_train_true, y_train_pred), mean_squared_error(y_train_true, y_train_pred), r2_score(y_train_true, y_train_pred)

# Final evaluation on test set
with torch.no_grad():
    y_test_pred, _ = model(X_tensor[val_index].to(device), seq_len_tensor[val_index].to(device))
    y_test_pred = y_test_pred.squeeze().cpu().numpy()
    y_test_true = y_tensor[val_index].cpu().numpy()
    mae_test, mse_test, r2_test = mean_absolute_error(y_test_true, y_test_pred), mean_squared_error(y_test_true, y_test_pred), r2_score(y_test_true, y_test_pred)

# Print final metrics for train and test sets
print(f"Train Metrics:")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MSE (Train): {mse_train:.4f}")
print(f"R² (Train): {r2_train:.4f}")

print("\nTest Metrics:")
print(f"MAE (Test): {mae_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Test): {r2_test:.4f}")

# Plot the loacss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), avg_train_losses, label='Train Loss')
plt.plot(range(num_epochs), avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# In[15]:
# Define 5-fold sequential splits
n_splits = 5
fold_size = len(X_tensor) // n_splits
r2_scores = []

# Sequentially use each fold as the validation set
for fold in range(n_splits):
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_splits - 1 else len(X_tensor)

    # Define training indices by excluding the current fold
    train_indices = np.concatenate((np.arange(0, val_start), np.arange(val_end, len(X_tensor))))
    val_indices = np.arange(val_start, val_end)

    # Sequentially split data into current train and validation fold
    X_train_fold = X_tensor[train_indices]
    y_train_fold = y_tensor[train_indices]
    seq_len_train_fold = seq_len_tensor[train_indices]

    X_val_fold = X_tensor[val_indices]
    y_val_fold = y_tensor[val_indices]
    seq_len_val_fold = seq_len_tensor[val_indices]

    # Create DataLoaders for the current fold
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold, seq_len_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold, seq_len_val_fold)

    train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer for each fold
    model = GRUAttentionModel(input_size=X_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers, attention_dim=attention_dim, output_size=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train model on the current training fold
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on the current validation fold
    model.eval()
    y_val_pred_fold, y_val_true_fold = [], []
    with torch.no_grad():
        for X_batch, y_batch, seq_len_batch in val_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            outputs, _ = model(X_batch, seq_len_batch)
            y_val_pred_fold.extend(outputs.squeeze().cpu().numpy())
            y_val_true_fold.extend(y_batch.cpu().numpy())

    # Calculate R² score for the current fold
    r2_fold = r2_score(y_val_true_fold, y_val_pred_fold)
    r2_scores.append(r2_fold)
    print(f"R² for fold {fold+1}: {r2_fold:.4f}")

# Print R² scores for all folds
print(f"R² scores for each fold: {r2_scores}")

# In[16]:
from sklearn.model_selection import KFold

# Initialize 5-fold cross-validation with random shuffling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

# Loop over each random fold
for fold, (train_indices, val_indices) in enumerate(kf.split(X_tensor)):
    # Define train and validation sets for the current fold
    X_train_fold = X_tensor[train_indices]
    y_train_fold = y_tensor[train_indices]
    seq_len_train_fold = seq_len_tensor[train_indices]

    X_val_fold = X_tensor[val_indices]
    y_val_fold = y_tensor[val_indices]
    seq_len_val_fold = seq_len_tensor[val_indices]

    # Create DataLoaders for the current fold
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold, seq_len_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold, seq_len_val_fold)

    train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer for each fold
    model = GRUAttentionModel(input_size=X_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers, attention_dim=attention_dim, output_size=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model on the current training fold
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch, seq_len_batch in train_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, seq_len_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on the current validation fold
    model.eval()
    y_val_pred_fold, y_val_true_fold = [], []
    with torch.no_grad():
        for X_batch, y_batch, seq_len_batch in val_loader:
            X_batch, y_batch, seq_len_batch = X_batch.to(device), y_batch.to(device), seq_len_batch.to(device)
            outputs, _ = model(X_batch, seq_len_batch)
            y_val_pred_fold.extend(outputs.squeeze().cpu().numpy())
            y_val_true_fold.extend(y_batch.cpu().numpy())

    # Calculate R² score for the current fold
    r2_fold = r2_score(y_val_true_fold, y_val_pred_fold)
    r2_scores.append(r2_fold)
    print(f"R² for fold {fold+1}: {r2_fold:.4f}")

# Print R² scores for all folds
print(f"R² scores for each fold: {r2_scores}")

# In[17]:
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the test.csv file containing test transcripts
test_data = pd.read_csv('Otest.csv')

# Filter out transcripts based on missing_test_ids
missing_test_ids = {4234, 4274, 4302, 4412, 4415, 4420, 4501}
valid_transcripts = test_data[~test_data['PARTID'].isin(missing_test_ids)].copy()

# Map the original indices of the valid transcripts for later use
valid_indices = valid_transcripts.index.tolist()

# Print debug information
print(f"Total transcripts in 'test_data': {len(test_data)}")
print(f"Total valid transcripts after filtering: {len(valid_transcripts)}")
print(f"Total valid indices: {len(valid_indices)}")
print(f"Total longitudinal_test samples: {len(longitudinal_test)}")


# Create a DataLoader for the valid transcripts
dataset = TensorDataset(X_test_tensor, test_seq_len_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Print the length of the DataLoader
print(f"Total batches in DataLoader after filtering: {len(loader)}")

# Store the predicted values for each transcript
predicted_values = []
attention_weights_all = []

# Function to get predictions and attention weights
with torch.no_grad():
    for i, (X_batch, seq_len_batch) in enumerate(loader):
        X_batch, seq_len_batch = X_batch.to(device), seq_len_batch.to(device)  # Move to device
        output, attention_weights = model(X_batch, seq_len_batch)
        original_idx = valid_indices[i]  # Map the DataLoader index to the original valid index
        predicted_values.append((output.item(), original_idx, attention_weights))
        attention_weights_all.append(attention_weights)

# Sort the predicted values in descending order to get the highest ranked transcripts
predicted_values.sort(reverse=True, key=lambda x: x[0])

# Select the top 5 highest ranked transcripts
top_5 = predicted_values[:5]

# Function to extract the highest-ranked window and its words
def extract_highest_attention_window_words(transcript_text, attention_weights, seq_len):
    attention_weights = attention_weights.squeeze()[:seq_len]  # Extract valid attention weights
    words = transcript_text.split()
    num_words_per_window = len(words) // len(attention_weights)
    max_idx = torch.argmax(attention_weights).item()
    start_idx = max_idx * num_words_per_window
    end_idx = min(start_idx + num_words_per_window, len(words))
    highest_attention_words = " ".join(words[start_idx:end_idx])
    return highest_attention_words

# Plot attention heatmaps for the top 5 transcripts with the highest attention window
for predicted_value, original_idx, attention_weights in top_5:
    transcript_text = valid_transcripts.iloc[original_idx]['text']
    transcript_id = valid_transcripts.iloc[original_idx]['PARTID']
    seq_len = test_seq_len_tensor[valid_indices.index(original_idx)].item()
    highest_attention_words = extract_highest_attention_window_words(transcript_text, attention_weights[0], seq_len)
    
    # Print the highest-ranked window words
    print(f"Transcript ID: {transcript_id}")
    print(f"Highest-ranked window words: {highest_attention_words}\n")

    # Plot the heatmap for visual inspection
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.squeeze()[:seq_len].cpu().unsqueeze(0).numpy(), cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)  # Add dimension to make it 2D
    plt.title(f'Attention Weights Heatmap for Transcript ID: {transcript_id}')
    plt.show()

# In[18]:
print(f"Size of X_test_tensor: {X_test_tensor.size()}")
print(f"Size of test_seq_len_tensor: {test_seq_len_tensor.size()}")
print(f"Size of y_test_tensor: {y_test_tensor.size()}")
