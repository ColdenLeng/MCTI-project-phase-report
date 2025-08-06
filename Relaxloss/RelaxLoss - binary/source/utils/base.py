import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import abc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
from .misc import Partition, get_all_losses, savefig
from .logger import AverageMeter, Logger
from .eval import accuracy, accuracy_binary
from progress.bar import Bar as Bar

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

__all__ = ['BaseTrainer']


class BaseTrainer(object):
    def __init__(self, the_args, save_dir):
        self.args = the_args
        self.save_dir = save_dir
        self.data_root = DATA_ROOT
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_logger()
        self.set_criterion()

    def set_cuda_device(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if hasattr(self.args, 'num_workers') and self.args.num_workers >= 1:
            torch.multiprocessing.set_start_method('spawn')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def set_seed(self):
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.random_seed)

    @abc.abstractmethod
    def set_dataloader(self):
        pass

    def set_logger(self):
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5', 'Val AUROC'])
        self.logger = logger

    def set_criterion(self):
        if getattr(self.args, 'binary', False):
            self.criterion = nn.BCEWithLogitsLoss()
            self.crossentropy = nn.BCEWithLogitsLoss()
            self.crossentropy_noreduce = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.crossentropy = nn.CrossEntropyLoss()
            self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')

    def train(self, model, optimizer, criterion):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        time_stamp = time.time()
       
        bar = Bar('Processing', max=len(self.trainloader), suffix='%(index)d/%(max)d | Loss: %(loss).4f | Acc: %(acc).2f%%')

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.float().unsqueeze(1)  # [B,1] for BCEWithLogitsLoss

            outputs = model(inputs)  # [B,1]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct = preds.eq(targets).sum().item()
            acc = 100.0 * correct / targets.size(0)

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc, inputs.size(0))

        
            bar.loss = losses.avg
            bar.acc = top1.avg
            bar.next()
        bar.finish()

        return losses.avg, top1.avg

    def test(self, model, criterion):
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
   
        bar = Bar('Processing', max=len(self.testloader), suffix='%(index)d/%(max)d | Loss: %(loss).4f | Acc: %(acc).2f%%')

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.float().unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct = preds.eq(targets).sum().item()
                acc = 100.0 * correct / targets.size(0)

                losses.update(loss.item(), inputs.size(0))
                top1.update(acc, inputs.size(0))
    
                bar.loss = losses.avg
                bar.acc = top1.avg
                bar.next()
        bar.finish()

        return losses.avg, top1.avg

    def get_loss_distributions(self, model):
        train_losses = get_all_losses(self.trainloader, model, self.crossentropy_noreduce, self.device)
        test_losses = get_all_losses(self.testloader, model, self.crossentropy_noreduce, self.device)
        return train_losses, test_losses

    def logger_plot(self):
        self.logger.plot(['train_loss', 'test_loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))
        self.logger.plot(['train_acc', 'test_acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))
