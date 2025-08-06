# generate_attack_data.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
from model import SmallCNN
from prepare_data import get_dataloaders

def get_attack_features(model, dataloader, device, num_classes=10):
    model.eval()
    features = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)

            # 1. softmax
            probs_np = probs.cpu().numpy()

            # 2. loss
            loss = F.nll_loss(log_probs, y, reduction='none').cpu().numpy().reshape(-1, 1)

            # 3. confidence
            confidence = torch.max(probs, dim=1)[0].cpu().numpy().reshape(-1, 1)

            # 4. correctness
            preds = torch.argmax(probs, dim=1)
            correctness = (preds == y).float().cpu().numpy().reshape(-1, 1)

            # 5. entropy
            entropy = (-probs * log_probs).sum(dim=1).cpu().numpy().reshape(-1, 1)

            # concat: softmax + [loss, conf, correct, entropy]
            batch_feat = np.hstack([probs_np, loss, confidence, correctness, entropy])
            features.append(batch_feat)

    return np.vstack(features)

def save_attack_data(model_path, dataloader, device, label, out_path):
    model = SmallCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    feat = get_attack_features(model, dataloader, device, num_classes=10)
    labels = np.full((feat.shape[0], 1), label)
    data = np.hstack([feat, labels])
    np.save(out_path, data)
    print(f"Saved: {out_path}, shape: {data.shape}")
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = get_dataloaders(data_root='../data')
    output_dir = '../results/attack_data'
    os.makedirs(output_dir, exist_ok=True)

    shadow_model = '../results/stl10/shadow/shadow_model.pth'
    target_model = '../results/stl10/vanilla/target_model.pth'

    save_attack_data(shadow_model, loaders['shadow_train'], device, 1, os.path.join(output_dir, 'shadow_train.npy'))
    save_attack_data(shadow_model, loaders['shadow_test'], device, 0, os.path.join(output_dir, 'shadow_test.npy'))
    save_attack_data(target_model, loaders['target_train'], device, 1, os.path.join(output_dir, 'target_train.npy'))
    save_attack_data(target_model, loaders['target_test'], device, 0, os.path.join(output_dir, 'target_test.npy'))


if __name__ == '__main__':
    main()
