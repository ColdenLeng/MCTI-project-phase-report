# attack-relaxloss/train_attack_model.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class AttackMLP(nn.Module):
    def __init__(self, input_size=10):
        super(AttackMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze()

def load_attack_data(file_list):
    X, y = [], []
    for path in file_list:
        data = np.load(path)
        X.append(data[:, :-1])
        y.append(data[:, -1])
    X = np.vstack(X)
    y = np.concatenate(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_attack_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def eval_attack_model(test_loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds)
    return auc

def main():
    os.makedirs("../results/relaxlossMIA", exist_ok=True)
    log_path = "../results/relaxlossMIA/attack_log.txt"
    log_file = open(log_path, "w")

    def log_and_print(msg):
        print(msg)                  
        log_file.write(msg + "\n") 
        log_file.flush()

    shadow_files = [
        '../results/attackDR/shadow_train.npy',
        '../results/attackDR/shadow_test.npy'
    ]
    target_files = [
        '../results/attackDR/target_train.npy',
        '../results/attackDR/target_test.npy'
    ]

    X_train, y_train = load_attack_data(shadow_files)
    X_test, y_test = load_attack_data(target_files)

    log_and_print(f"Detected feature dim: {X_test.shape[1]}")

    if X_test.shape[1] > 10:
        loss_auc = roc_auc_score(y_test.numpy(), -X_test[:, 10].numpy())
        log_and_print(f"Loss Attack AUC (negated): {loss_auc:.4f}")
    else:
        log_and_print("Loss Attack AUC: [N/A] (missing loss feature)")

    if X_test.shape[1] > 11:
        conf_auc = roc_auc_score(y_test.numpy(), X_test[:, 11].numpy())
        log_and_print(f"Confidence Attack AUC: {conf_auc:.4f}")
    else:
        log_and_print("Confidence Attack AUC: [N/A] (missing confidence feature)")

    softmax = X_test[:, :10].numpy()
    softmax_max = np.max(softmax, axis=1)
    softmax_auc = roc_auc_score(y_test.numpy(), softmax_max)
    log_and_print(f"Softmax Max Attack AUC: {softmax_auc:.4f}")

    log_file.close()


if __name__ == '__main__':
    main()
