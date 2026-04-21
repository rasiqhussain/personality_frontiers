# In[1]:
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt

# Load embeddings and labels
with open('mean_train', 'rb') as handle:
    mean_train = pickle.load(handle)

with open('mean_test', 'rb') as handle:
    mean_test = pickle.load(handle)

with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)

with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(mean_train, dtype=torch.float32)
X_test_tensor = torch.tensor(mean_test, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.float32)

# Combine train and test for cross-validation
X = torch.cat((X_train_tensor, X_test_tensor), dim=0)
y = torch.cat((y_train_tensor, y_test_tensor), dim=0)

# Ensure the input size matches the expected shape
input_size = X.shape[1]

# Define the Feedforward Neural Network with Dropout
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
        out = self.fc1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# Define the objective function for Optuna using MAE
def objective(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 30, 100)

    model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1)
    criterion = nn.L1Loss()  # Use L1Loss for MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    avg_val_r2 = 0.0
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val).squeeze()
            val_r2 = r2_score(y_val.numpy(), y_val_pred.numpy())
            avg_val_r2 += val_r2 / 5.0

    return avg_val_r2

# Run Optuna for hyperparameter tuning
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train and evaluate the final model with the best hyperparameters
hidden_size1 = best_params['hidden_size1']
hidden_size2 = best_params['hidden_size2']
learning_rate = best_params['learning_rate']
num_epochs = best_params['num_epochs']
weight_decay = best_params['weight_decay']

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_train_losses, all_val_losses = [], []
all_mae_train, all_mae_val = [], []
all_mse_train, all_mse_val = [], []
all_r2_train, all_r2_val = [], []
all_r_train, all_r_val = [], []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1)
    criterion = nn.L1Loss()  # Use L1Loss for MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    mae_train_scores, mae_val_scores = [], []
    mse_train_scores, mse_val_scores = [], []
    r2_train_scores, r2_val_scores = [], []
    r_train_scores, r_val_scores = [], []

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val).squeeze()
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())

            y_train_pred = model(X_train).squeeze()
            y_train_np = y_train.numpy()
            y_train_pred_np = y_train_pred.numpy()
            mae_train = mean_absolute_error(y_train_np, y_train_pred_np)
            mse_train = mean_squared_error(y_train_np, y_train_pred_np)
            r2_train = r2_score(y_train_np, y_train_pred_np)
            correlation_matrix_train = np.corrcoef(y_train_np, y_train_pred_np)
            correlation_xy_train = correlation_matrix_train[0, 1]
            r_train = correlation_xy_train

            y_val_np = y_val.numpy()
            y_val_pred_np = y_val_pred.numpy()
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

# Print averaged results for the last epoch
print(f'Average Train Loss: {avg_train_losses[-1]:.4f}')
print(f'Average Val Loss: {avg_val_losses[-1]:.4f}')
print(f'Average MAE (Train): {np.mean([mae[-1] for mae in all_mae_train]):.4f}')
print(f'Average MSE (Train): {np.mean([mse[-1] for mse in all_mse_train]):.4f}')
print(f'Average R² (Train): {np.mean([r2[-1] for r2 in all_r2_train]):.4f}')
print(f'Average Correlation coefficient (r) (Train): {np.mean([r[-1] for r in all_r_train]):.4f}')
print(f'Average MAE (Val): {np.mean([mae[-1] for mae in all_mae_val]):.4f}')
print(f'Average MSE (Val): {np.mean([mse[-1] for mse in all_mse_val]):.4f}')
print(f'Average R² (Val): {np.mean([r2[-1] for r2 in all_r2_val]):.4f}')
print(f'Average Correlation coefficient (r) (Val): {np.mean([r[-1] for r in all_r_val]):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'feedforward_nn_model.pth')

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
plt.plot(range(num_epochs), np.mean(all_mae_train, axis=0), label='Train MAE')
plt.plot(range(num_epochs), np.mean(all_mae_val, axis=0), label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE Over Epochs')
plt.show()

# Plot R² over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), np.mean(all_r2_train, axis=0), label='Train R²')
plt.plot(range(num_epochs), np.mean(all_r2_val, axis=0), label='Validation R²')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.legend()
plt.title('R² Over Epochs')
plt.show()
