import os
import sys
import argparse
import random
import shutil
import numpy as np
import torch
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/vanilla')
from base import NonImageTrainer
import models as models
from utils import mkdir, str2bool, load_yaml, write_yaml, adjust_learning_rate


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, help='experiment name, used for set up save_dir')
    parser.add_argument('--dataset', type=str, choices=['Texas', 'Purchase', 'Covertype', 'Adult'], help='dataset name')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--schedule_milestone', type=int, nargs='+', help='when to decrease the learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate step gamma')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--train_batchsize', type=int, help='training batch size')
    parser.add_argument('--test_batchsize', type=int, help='testing batch size')
    parser.add_argument('--num_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--partition', type=str, choices=['target', 'shadow'], help='training partition')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    parser.add_argument('--binary', type=str2bool, default=False, help='Use BCEWithLogitsLoss for binary classification')
    return parser


def check_args(parser):
    args = parser.parse_args()
    save_dir = os.path.join(SAVE_ROOT % args.dataset, args.exp_name)
    if args.partition == 'shadow':
        save_dir = os.path.join(save_dir, 'shadow')
    mkdir(save_dir)
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10 ^ 5)

    if args.if_onlyeval:
        preload_configs = load_yaml(os.path.join(save_dir, 'params.yml'))
        parser.set_defaults(**preload_configs)
        args = parser.parse_args()
    else:
        default_configs = load_yaml(FILE_DIR + '/configs/default.yml')
        parser.set_defaults(**default_configs)
        args = parser.parse_args()
        write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def main():
    args, save_dir = check_args(parse_arguments())

    trainer = NonImageTrainer(args, save_dir)
    trainer.logger.names = ['lr', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_acc5', 'test_acc5']
    for name in trainer.logger.names:
        trainer.logger.numbers[name] = []

    model = models.__dict__[args.dataset]()
    model = torch.nn.DataParallel(model)
    model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    criterion = torch.nn.BCEWithLogitsLoss() if args.binary else torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    logger = trainer.logger

    if args.if_resume or args.if_onlyeval:
        try:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pkl'))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
        except:
            pass

    if args.if_onlyeval:
        print('\nEvaluation only')
        test_loss, test_acc, test_acc5 = trainer.test(model)
        print(' Test Loss: %.8f, Test Acc(top1): %.2f, Test Acc(top5): %.2f' % (test_loss, test_acc, test_acc5))
        return

    for epoch in range(start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.gamma, args.schedule_milestone)
        train_loss, train_acc = trainer.train(model, optimizer, criterion)
        test_loss, test_acc = trainer.test(model, criterion)
        test_acc5 = 0.0  

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, train_loss, test_loss, train_acc, test_acc, test_acc5, test_acc5])
    
        print('Epoch %d, Train acc: %f, Test acc: %f, lr: %f' % (epoch, train_acc, test_acc, lr))

        save_dict = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

        trainer.logger_plot()


if __name__ == '__main__':
    main()

