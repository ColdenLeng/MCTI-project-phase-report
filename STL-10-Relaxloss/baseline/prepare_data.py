# baseline/prepare_data.py

import numpy as np
import torch
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import Subset

def get_dataloaders(data_root, batch_size=64, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = STL10(root=data_root, split='train', download=False, transform=transform)

    # split shadow and vanilla
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    target_idx = indices[:2500]
    shadow_idx = indices[2500:]

    target_train = Subset(dataset, target_idx[:1250])
    target_test  = Subset(dataset, target_idx[1250:])
    shadow_train = Subset(dataset, shadow_idx[:1250])
    shadow_test  = Subset(dataset, shadow_idx[1250:])

    def loader(subset): return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

    return {
        'target_train': loader(target_train),
        'target_test':  loader(target_test),
        'shadow_train': loader(shadow_train),
        'shadow_test':  loader(shadow_test),
    }
