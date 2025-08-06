# relaxloss/train_target_relaxloss.py

import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
from model import SmallCNN
from prepare_data import get_dataloaders

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels].to(labels.device)

def accuracy(output, target, topk=(1,)):
    """Computes the top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # topk scores, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0).item() * 100. / batch_size for k in topk]

class RelaxTrainer:
    def __init__(self, model, loaders, device, alpha=2.0, upper=0.6, num_epochs=60, lr=0.1, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.trainloader = loaders['target_train']
        self.testloader = loaders['target_test']
        self.alpha = alpha
        self.upper = upper
        self.epochs = num_epochs
        self.criterion_ce = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 45], gamma=0.1)

    def train(self):
        log_lines = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total = 0

            for x, y in tqdm(self.trainloader, desc=f"[Train] Epoch {epoch+1}/{self.epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_ce_full = self.criterion_ce(logits, y)
                loss_ce = loss_ce_full.mean()

                if epoch % 2 == 0:
                    loss = (loss_ce - self.alpha).abs()
                else:
                    if loss_ce > self.alpha:
                        loss = loss_ce
                    else:
                        pred = torch.argmax(logits, dim=1)
                        correct = torch.eq(pred, y).float()
                        probs = F.softmax(logits, dim=1)
                        conf_target = probs[torch.arange(y.size(0)), y]
                        conf_target = torch.clamp(conf_target, max=self.upper)
                        conf_else = (1.0 - conf_target) / (logits.size(1) - 1)
                        onehot = one_hot_embedding(y, num_classes=logits.size(1))
                        soft_target = onehot * conf_target.unsqueeze(1) + (1 - onehot) * conf_else.unsqueeze(1)
                        log_probs = F.log_softmax(logits, dim=1)
                        soft_ce = -torch.sum(soft_target * log_probs, dim=1)
                        loss = (1 - correct) * soft_ce - 1.0 * loss_ce_full
                        loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            self.scheduler.step()
            train_acc = total_correct / total
            test_acc = self.evaluate()
            log_line = f"Epoch {epoch+1}/{self.epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
            print(log_line)
            log_lines.append(log_line)

        return log_lines

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = get_dataloaders(data_root='../data')
    model = SmallCNN(num_classes=10)

    with open('relaxloss_config.yml', 'r') as f:
        config = yaml.safe_load(f)

    alpha = config['relaxloss']['alpha']
    upper = config['relaxloss']['upper']
    epochs = config['relaxloss']['epochs']
    lr = config['relaxloss']['lr']
    weight_decay = config['relaxloss']['weight_decay']

    trainer = RelaxTrainer(
        model, loaders, device,
        alpha=alpha,
        upper=upper,
        num_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay
    )

    logs = trainer.train()

    save_dir = '../results/stl10/relaxloss'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'relaxloss_model.pth'))

    with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
        for line in logs:
            f.write(line + '\n')

    print(f"Model + log saved to {save_dir}")

if __name__ == '__main__':
    main()
