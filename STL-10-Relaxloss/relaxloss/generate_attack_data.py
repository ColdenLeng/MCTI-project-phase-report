# attack-relaxloss/generate_attack_data.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
from model import SmallCNN
from prepare_data import get_dataloaders

def get_softmax_outputs(model, dataloader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Extracting"):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.vstack(outputs)

def save_target_data(model_path, dataloader, device, label, out_path):
    model = SmallCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_features = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting features"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction='none')         # (batch,)
            confidence = probs.max(dim=1).values                        # (batch,)
            correctness = (logits.argmax(dim=1) == y).float()           # (batch,)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)     # (batch,)

            features = torch.cat([
                probs,
                loss.unsqueeze(1),
                confidence.unsqueeze(1),
                correctness.unsqueeze(1),
                entropy.unsqueeze(1)
            ], dim=1)

            labels = torch.full((x.size(0), 1), label, device=device)
            full = torch.cat([features, labels], dim=1)

            all_features.append(full.cpu())

    final = torch.cat(all_features, dim=0).numpy()
    np.save(out_path, final)
    print(f"Saved: {out_path}, shape: {final.shape}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = get_dataloaders(data_root='../data')

    output_dir = '../results/attackDR'
    os.makedirs(output_dir, exist_ok=True)

    # Target Model (RelaxLoss)
    save_target_data(
        model_path='../results/stl10/relaxloss/relaxloss_model.pth',
        dataloader=loaders['target_train'],
        device=device,
        label=1,
        out_path=os.path.join(output_dir, 'target_train.npy')
    )

    save_target_data(
        model_path='../results/stl10/relaxloss/relaxloss_model.pth',
        dataloader=loaders['target_test'],
        device=device,
        label=0,
        out_path=os.path.join(output_dir, 'target_test.npy')
    )

    # Shadow Model (Baseline)
    save_target_data(
        model_path='../results/stl10/shadow/shadow_model.pth',
        dataloader=loaders['shadow_train'],
        device=device,
        label=1,
        out_path=os.path.join(output_dir, 'shadow_train.npy')
    )

    save_target_data(
        model_path='../results/stl10/shadow/shadow_model.pth',
        dataloader=loaders['shadow_test'],
        device=device,
        label=0,
        out_path=os.path.join(output_dir, 'shadow_test.npy')
    )

if __name__ == '__main__':
    main()

