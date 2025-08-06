import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms
from matplotlib.pyplot import savefig

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../../data')
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
from dataset import CIFAR10, CIFAR100
from dataset import STL10
from utils import BaseTrainer, Partition


class CIFARTrainer(BaseTrainer):
    def set_dataloader(self):
        """The function to set the dataset parameters"""

        if self.args.dataset == 'CIFAR10':
            self.dataset = CIFAR10
            self.num_classes = 10
            self.dataset_size = 60000
        elif self.args.dataset == 'CIFAR100':
            self.dataset = CIFAR100
            self.num_classes = 100
            self.dataset_size = 60000
        elif self.args.dataset == 'STL10':
            self.dataset = STL10
            self.num_classes = 10
            self.dataset_size = 5000  # STL10 train set size
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        if self.args.if_data_augmentation:
            print('With data augmentation')
            if self.args.dataset in ['CIFAR10', 'CIFAR100']:
                transform_train = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))])
            elif self.args.dataset == 'STL10':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5))])
        else:
            print('Without data augmentation')
            if self.args.dataset in ['CIFAR10', 'CIFAR100']:
                transform_train = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))])
            elif self.args.dataset == 'STL10':
                transform_train = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5))])

        
        if self.args.dataset in ['CIFAR10', 'CIFAR100']:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))])
        elif self.args.dataset == 'STL10':
            transform_test = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])

        self.transform_train = transform_train
        self.transform_test = transform_test

        if self.args.partition == 'target':
            indices = np.arange(self.dataset_size).astype(int)
            np.random.shuffle(indices)
            np.save(os.path.join(self.save_dir, 'full_idx'), indices)
            partition = Partition(dataset_size=self.dataset_size, indices=indices)
            self.partition = partition
            self.trainset_idx, self.testset_idx = partition.get_target_indices()
        elif self.args.partition == 'shadow':
            try:
                target_path = os.path.join(self.save_dir.replace("shadow", ""), 'full_idx.npy')
                indices = np.load(target_path)
                print('Load indices from target model:', target_path)
            except:
                print('Cannot find target model, reinitialize indices')
                indices = np.arange(self.dataset_size).astype(int)
                np.random.shuffle(indices)
                np.save(os.path.join(self.save_dir, 'full_idx'), indices)
            partition = Partition(dataset_size=self.dataset_size, indices=indices)
            self.partition = partition
            self.trainset_idx, self.testset_idx = partition.get_shadow_indices()
        else:
            raise ValueError("args.partition must be 'target' or 'shadow'")

        trainset = self.dataset(root=self.data_root, indices=self.trainset_idx,
                                download=True, transform=self.transform_train)
        testset = self.dataset(root=self.data_root, indices=self.testset_idx,
                               download=True, transform=self.transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batchsize,
                                                  shuffle=True, num_workers=self.args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batchsize,
                                                 shuffle=False, num_workers=self.args.num_workers)
        self.trainset = trainset
        self.trainloader = trainloader
        self.testset = testset
        self.testloader = testloader
        
    def logger_plot(self):
        """Override logger_plot to match image-specific logger names"""
        self.logger.plot(['Train CE', 'Test CE'])  
        savefig(os.path.join(self.save_dir, 'loss.png'))
    
        self.logger.plot(['Train Acc', 'Test Acc'])  
        savefig(os.path.join(self.save_dir, 'acc.png'))
