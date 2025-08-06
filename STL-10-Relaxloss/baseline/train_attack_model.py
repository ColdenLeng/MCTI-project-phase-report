# attack/train_attack_model.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class SimpleBenchmark:
    def __init__(self, s_tr, s_te, t_tr, t_te):
        self.s_tr = s_tr
        self.s_te = s_te
        self.t_tr = t_tr
        self.t_te = t_te

    def _thre_setting(self, tr, te):
        vals = np.concatenate((tr, te))
        best_acc, best_thre = 0, None
        for v in vals:
            acc = 0.5 * (np.sum(tr >= v) / len(tr) + np.sum(te < v) / len(te))
            if acc > best_acc:
                best_acc, best_thre = acc, v
        return best_thre

    def acc_report(self, name):
        thre = self._thre_setting(self.s_tr, self.s_te)
        mem = np.sum(self.t_tr >= thre) / len(self.t_tr)
        non = np.sum(self.t_te < thre) / len(self.t_te)
        acc = 0.5 * (mem + non)
        print(f"MIA via {name} (general threshold): the attack acc is {acc:.3f}")

    def auc_report(self, name):
        labels = np.concatenate((np.zeros_like(self.t_te), np.ones_like(self.t_tr)))
        values = np.concatenate((self.t_te, self.t_tr))
        auc = metrics.roc_auc_score(labels, values)
        ap = metrics.average_precision_score(labels, values)
        print(f"MIA via {name} AUC: the attack auc is {auc:.3f}, ap is {ap:.3f}")

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
        X.append(data[:, :-1])  # features
        y.append(data[:, -1])   # member labels
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
    acc = accuracy_score(labels, preds >= 0.5)
    f1 = f1_score(labels, preds >= 0.5)
    return auc, acc, f1
    
def single_feature_auc(X, y, index_or_values, name="", negate=False):
    if isinstance(index_or_values, int):  
        feature = X[:, index_or_values].numpy()
    else:
        
        if hasattr(index_or_values, "numpy"):
            feature = index_or_values.numpy()
        else:
            feature = index_or_values  
    if negate:
        feature = -feature
    auc = roc_auc_score(y.numpy(), feature)
    print(f"{name:<22} AUC: {auc:.4f}")
    
def blackbox_summary(X_test, y_test, name, negate=False):
    
    idx = {'loss': 10, 'confidence': 11, 'entropy': 13}  
    feat = X_test[:, idx[name]].numpy() if hasattr(X_test, 'numpy') else X_test[:, idx[name]]
    if negate:
        feat = -feat
    t_tr = feat[y_test == 1]
    t_te = feat[y_test == 0]
    bench = SimpleBenchmark(t_tr, t_te, t_tr, t_te)
    bench.acc_report(f'{name} ACC')
    bench.auc_report(f'{name}')
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    shadow_files = [
        '../results/attack_data/shadow_train.npy',
        '../results/attack_data/shadow_test.npy'
    ]
    X_train, y_train = load_attack_data(shadow_files)

    target_files = [
        '../results/attack_data/target_train.npy',
        '../results/attack_data/target_test.npy'
    ]
    X_test, y_test = load_attack_data(target_files)
    
    print("\n[Single-Feature Attack AUCs]")
    single_feature_auc(X_test, y_test, 10, "Loss (negated)", negate=True)
    single_feature_auc(X_test, y_test, 11, "Confidence")
    single_feature_auc(X_test, y_test, 12, "Correctness")
    single_feature_auc(X_test, y_test, 13, "Entropy (negated)", negate=True)
    single_feature_auc(X_test, y_test, np.argmax(X_test[:,:10].numpy(), axis=1), "Softmax Max Class")
    
    softmax_max_conf = torch.from_numpy(np.max(X_test[:, :10].numpy(), axis=1))
    single_feature_auc(X_test, y_test, softmax_max_conf, "Softmax Max Conf")
    
    
    shadow_files = [
        '../results/attack_data/shadow_train.npy',
        '../results/attack_data/shadow_test.npy'
    ]
    X_train, y_train = load_attack_data(shadow_files)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = AttackMLP(input_size=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    auc_list = []
    best_auc = 0.0
    num_epochs = 10

    for epoch in range(num_epochs):
        train_attack_model(train_loader, model, criterion, optimizer, device)
        auc, acc, f1 = eval_attack_model(test_loader, model, device)
        auc_list.append(auc)
        best_auc = max(best_auc, auc)

        #print(f"[Epoch {epoch+1}/{num_epochs}] AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        if epoch == 0: 
            print("\n[DEBUG] Sample feature vector:")
            #print("Softmax:", X_train[0][:10].numpy())
            print("Loss:", X_train[0][10].item())
            print("Confidence:", X_train[0][11].item())
            print("Correctness:", X_train[0][12].item())
            print("Entropy:", X_train[0][13].item())

            print("\n[DEBUG] Feature stats (train set):")
            print(f"Loss mean: {X_train[:,10].mean():.4f} | std: {X_train[:,10].std():.4f}")
            print(f"Conf mean: {X_train[:,11].mean():.4f} | std: {X_train[:,11].std():.4f}")
            print(f"Entropy mean: {X_train[:,13].mean():.4f} | std: {X_train[:,13].std():.4f}")

    final_auc = auc_list[-1]
    avg_auc = sum(auc_list) / len(auc_list)

    print(f"\nFinal Attack AUC on Target: {final_auc:.4f}")
    print(f"Best  Attack AUC on Target: {best_auc:.4f}")
    print(f"Avg.  Attack AUC over {num_epochs} epochs: {avg_auc:.4f}")
    
    print("\n[Fixed Single-Feature AUCs]")
    print("Loss (neg):", roc_auc_score(y_test, -X_test[:,10].numpy()))
    print("Entropy (neg):", roc_auc_score(y_test, -X_test[:,13].numpy()))
    
    

    

if __name__ == '__main__':
    main()

    

