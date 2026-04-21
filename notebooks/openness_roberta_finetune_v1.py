# In[1]:
# Load embeddings, labels, and sequence lengths
import pickle

with open('mean_train', 'rb') as handle:
    mean_train = pickle.load(handle)
with open('labels', 'rb') as handle:
    labels = pickle.load(handle)
with open('train_seq_len', 'rb') as handle:
    train_seq_len = pickle.load(handle)

# Display loaded data
print("Mean Train Sample:", mean_train[:2])  # Display first 2 samples for brevity
print("Labels Sample:", labels[:2])  # Display first 2 labels
print("Train Seq Length Sample:", train_seq_len[:2])  # Display first 2 sequence lengths

# In[2]:
# Display additional samples to get a better understanding
print("Additional Mean Train Samples (first 3):")
for i, sample in enumerate(mean_train[:3]):
    print(f"Sample {i+1} shape: {sample.shape}")
    print(sample)

print("\nAdditional Labels Samples (first 3):")
print(labels.head(3))  # Display the first 3 rows

print("\nTrain Seq Lengths (first 3):")
print(train_seq_len[:3])  # Display the first 3 sequence lengths

# Check shapes of loaded variables
print("\nShapes of Loaded Variables:")
print("Mean Train shape (overall structure):", len(mean_train), "sequences with individual shapes varying per sequence.")
print("Labels shape:", labels.shape)
print("Train Seq Length shape:", train_seq_len.shape)

# In[3]:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load data (replace `X` and `y` with your loaded embeddings and labels)
input_size = mean_train.shape[1]  # assuming each embedding is a vector of fixed size
X = torch.tensor(mean_train, dtype=torch.float32)
y = torch.tensor(labels['labels'].values, dtype=torch.float32).view(-1, 1)

# Split into train-test (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective function using MSE
def objective(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 30, 100)

    model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        mse_val = mean_squared_error(y_test.numpy(), y_test_pred.numpy())
    
    return mse_val

# Run Optuna for hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train and evaluate the final model with best parameters
hidden_size1 = best_params['hidden_size1']
hidden_size2 = best_params['hidden_size2']
learning_rate = best_params['learning_rate']
num_epochs = best_params['num_epochs']
weight_decay = best_params['weight_decay']

model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        val_loss = criterion(y_test_pred, y_test)
        val_losses.append(val_loss.item())

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# In[4]:
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load data (replace `X` and `y` with your actual embeddings and labels)
input_size = mean_train.shape[1]  # assuming each embedding is a vector of fixed size
X = torch.tensor(mean_train, dtype=torch.float32)
y = torch.tensor(labels['labels'].values, dtype=torch.float32).view(-1, 1)

# Split data into training and validation sets (80-20 split)
train_size = int(0.6 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Best hyperparameters
hidden_size1 = 215
hidden_size2 = 60
learning_rate = 0.008838787433819049
weight_decay = 1.9265744263063342e-05
num_epochs = 90
# Initialize model, loss function, and optimizer
model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1)
criterion = nn.MSELoss()  # MSE as the loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Lists to track loss over epochs
train_losses, val_losses = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = criterion(y_val_pred, y_val)
        val_losses.append(val_loss.item())

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Final Evaluation
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val).squeeze()
    y_train_pred = model(X_train).squeeze()
    mse_val = mean_squared_error(y_val.numpy(), y_val_pred.numpy())
    mae_val = mean_absolute_error(y_val.numpy(), y_val_pred.numpy())
    r2_val = r2_score(y_val.numpy(), y_val_pred.numpy())

    mse_train = mean_squared_error(y_train.numpy(), y_train_pred.numpy())
    mae_train = mean_absolute_error(y_train.numpy(), y_train_pred.numpy())
    r2_train = r2_score(y_train.numpy(), y_train_pred.numpy())

print(f'Final Training MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}')
print(f'Final Validation MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}')

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()
