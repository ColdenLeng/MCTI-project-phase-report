# baseline/train_target.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from model import SmallCNN
from prepare_data import get_dataloaders

def train(model, dataloader, criterion, optimizer, device, epoch_log):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = correct / total
    epoch_log['train_acc'].append(acc)
    return total_loss, acc

def evaluate(model, dataloader, device, epoch_log):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    epoch_log['test_acc'].append(acc)
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loaders = get_dataloaders(data_root='../data')
    train_loader = loaders['target_train']
    test_loader  = loaders['target_test']

    model = SmallCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    result_dir = '../results/stl10/vanilla'
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'log.txt')
    model_path = os.path.join(result_dir, 'target_model.pth')

    epoch_log = {'train_acc': [], 'test_acc': []}
    
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    lr = config['train']['lr']

    for epoch in range(epochs):#yml train epochs
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch_log)
        test_acc = evaluate(model, test_loader, device, epoch_log)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\n")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log saved to {log_path}")

if __name__ == '__main__':
    main()
