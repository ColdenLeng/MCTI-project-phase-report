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
    """The class that contains the code for base trainer class."""

    def __init__(self, the_args, save_dir):
        """The function to initialize this class."""
        self.args = the_args
        self.save_dir = save_dir
        self.data_root = DATA_ROOT
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_logger()
        self.set_criterion()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if hasattr(self.args, 'num_workers') and self.args.num_workers >= 1:
            torch.multiprocessing.set_start_method('spawn')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def set_seed(self):
        """Set random seed"""
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.random_seed)

    @abc.abstractmethod
    def set_dataloader(self):
        """The function to set the dataset parameters"""
        self.dataset = None
        self.num_classes = None
        self.dataset_size = None
        self.transform_train = None
        self.transform_test = None

        self.partition = None
        self.trainset_idx = None
        self.testset_idx = None

        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5', 'Val AUROC'])
        self.logger = logger

    def set_criterion(self):
        """Set up criterion"""
        self.criterion = nn.CrossEntropyLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')

    def train(self, model, optimizer, *args):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()
        
        all_probs = []          
        all_targets = []        

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

        bar.finish()
        
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        
        y_true = label_binarize(all_targets, classes=np.arange(self.num_classes))

        
        auroc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')
        print(f"Test AUROC (macro avg): {auroc:.4f}")
        return (losses.avg, top1.avg, top5.avg)

    def test(self, model):
        """Test"""
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        model.eval()
        criterion = self.crossentropy
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        all_probs = []
        all_targets = []

        bar = Bar('Processing', max=len(self.testloader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.long()
    
                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)
    
                ### Forward
                outputs = model(inputs)
    
                ### Save for AUROC
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

                ### Evaluate
                loss = criterion(outputs, targets)
                #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                if outputs.size(1) < 5:
                    prec1, = accuracy(outputs.data, targets.data, topk=(1,))
                    prec5 = torch.tensor(0.0).to(outputs.device)  # dummy value for logging
                else:
                    prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                    
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
    
                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()
    
                ### Progress bar
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.testloader),
                    data=dataload_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        bar.finish()

        ### AUROC Calculation (safe version)
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # AUROC calculation with class presence check
        present_classes = np.unique(all_targets)
        print(f"[DEBUG] AUROC y_true classes: {np.unique(all_targets, return_counts=True)}")

        try:
            # One-hot encode for only the present classes
            y_true = label_binarize(all_targets, classes=present_classes)

            # Slice only the present class probs
            y_probs = all_probs[:, present_classes]

            # Compute AUROC
            auroc = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
            print(f"Test AUROC (macro avg): {auroc:.4f}")

        except Exception as e:
            print(f"[Warning] AUROC calculation skipped: {e}")
            auroc = -1.0

        return (losses.avg, top1.avg, top5.avg, auroc)

    def get_loss_distributions(self, model):
        """ Obtain the member and nonmember loss distributions"""
        train_losses = get_all_losses(self.trainloader, model, self.crossentropy_noreduce, self.device)
        test_losses = get_all_losses(self.testloader, model, self.crossentropy_noreduce, self.device)
        return train_losses, test_losses

    def logger_plot(self):
        """ Visualize the training progress"""
        #self.logger.plot(['Train Loss', 'Val Loss'])
        self.logger.plot(['train_loss', 'test_loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))

        #self.logger.plot(['Train Acc', 'Val Acc'])
        self.logger.plot(['train_acc', 'test_acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))


