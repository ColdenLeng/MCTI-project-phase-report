import os
import sys
import torch
import numpy as np
import urllib.request
import tarfile
import pandas as pd
from scipy import sparse
from collections import Counter

   
from sklearn.model_selection import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
sys.path.append(os.path.join(FILE_DIR, '../'))
from utils import mkdir
from sklearn.model_selection import train_test_split


def create_tensor_dataset(features, labels):
    """Create TensorDataset"""
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features])  # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset


def prepare_texas(seed=1000, if_avg_split=True):
    '''
    Texas dataset:
    X: (67330,6169) binary feature
    Y: (67330,)  num_classes=100
    '''

    ## Dataset directory
    DATASET_PATH = os.path.join(DATA_ROOT, 'Texas')
    mkdir(DATASET_PATH)
    DATASET_FEATURES = os.path.join(DATASET_PATH, 'texas', '100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH, 'texas', '100/labels')
    DATASET_NUMPY = 'data.npz'

    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    if not os.path.isfile(os.path.join(DATASET_PATH, DATASET_NUMPY)):
        print('Creating data.npz file from raw data')
        data_set_features = np.genfromtxt(DATASET_FEATURES, delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS, delimiter=',')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32) - 1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    ## Load data
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    r = np.load(os.path.join(DATA_ROOT, 'dataset_shuffle/random_r_texas100.npy'))
    X = X[r]
    Y = Y[r]
    len_train = len(X)

    ## Split data
    if if_avg_split:  # split evenly
        train_classifier_ratio, train_attack_ratio = 0.4, 0.2
    else:  # using fixed number of samples to train target model
        train_classifier_ratio, train_attack_ratio = float(10000) / float(X.shape[0]), 0.3
    train_data = X[:int(train_classifier_ratio * len_train)]
    test_data = X[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    train_attack_data = X[int(train_classifier_ratio * len_train):int(
        (train_classifier_ratio + train_attack_ratio) * len_train)]

    train_label = Y[:int(train_classifier_ratio * len_train)]
    test_label = Y[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    train_attack_label = Y[int(train_classifier_ratio * len_train):int(
        (train_classifier_ratio + train_attack_ratio) * len_train)]

    ## Generate shadow and target partition
    np.random.seed(seed)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len // 2]
    target_indices = np.delete(np.arange(train_len), shadow_indices)
    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1 * train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len // 2]
    target_indices = np.delete(np.arange(test_len), shadow_indices)
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    ## Generate dataloader
    shadow_train = create_tensor_dataset(shadow_train_data, shadow_train_label)
    shadow_test = create_tensor_dataset(shadow_test_data, shadow_test_label)

    target_train = create_tensor_dataset(target_train_data, target_train_label)
    target_test = create_tensor_dataset(target_test_data, target_test_label)

    pseudoattack_data = create_tensor_dataset(train_attack_data, train_attack_label)
    return target_train, target_test, shadow_train, shadow_test, pseudoattack_data


def prepare_purchase(seed=1000, if_avg_split=True):
    '''
    purchase
    X: (197324, 600) binary feature
    Y: (197324,)  100 classes
    '''
    DATASET_PATH = os.path.join(DATA_ROOT, 'Purchase')
    mkdir(DATASET_PATH)
    DATASET_NAME = 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'
    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    if not os.path.isfile(os.path.join(DATASET_PATH, DATASET_NUMPY)):
        print('Creating data.npz file from raw data')
        data_set = np.genfromtxt(DATASET_FILE, delimiter=',')
        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    ## Load data
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    r = np.load(os.path.join(DATA_ROOT, 'dataset_shuffle/random_r_purchase100.npy'))
    X = X[r]
    Y = Y[r]
    len_train = len(X)

    ## Split data
    if if_avg_split:
        '''
        target size: 39465(train), 39465(test)
        shadow size: 39464(train), 39464(test)
        pseudoattack size: 39465
        '''
        train_classifier_ratio, train_attack_ratio = 0.4, 0.2
    else:
        '''
        target size: 9866(train), 9866(test)
        shadow size: 9866(train), 9866(test)
        '''
        train_classifier_ratio, train_attack_ratio = 0.1, 0.3
    train_data = X[:int(train_classifier_ratio * len_train)]
    test_data = X[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    train_attack_data = X[int(train_classifier_ratio * len_train):int(
        (train_classifier_ratio + train_attack_ratio) * len_train)]

    train_label = Y[:int(train_classifier_ratio * len_train)]
    test_label = Y[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    train_attack_label = Y[int(train_classifier_ratio * len_train):int(
        (train_classifier_ratio + train_attack_ratio) * len_train)]

    ## Generate shadow and target partition
    np.random.seed(seed)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len // 2]
    target_indices = r[train_len // 2:]
    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1 * train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len // 2]
    target_indices = r[test_len // 2:]
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    shadow_train = create_tensor_dataset(shadow_train_data, shadow_train_label)
    shadow_test = create_tensor_dataset(shadow_test_data, shadow_test_label)
    target_train = create_tensor_dataset(target_train_data, target_train_label)
    target_test = create_tensor_dataset(target_test_data, target_test_label)
    pseudoattack_data = create_tensor_dataset(train_attack_data, train_attack_label)
    return target_train, target_test, shadow_train, shadow_test, pseudoattack_data
'''
def prepare_mimic_sparse(seed=1000, if_avg_split=True, sparsity_threshold=20):
    """
    MIMIC dataset (sparse-aware version):
    X: (n_samples, n_features) - CSR sparse matrix
    Y: (n_samples,) - np.ndarray
    """

    print("[INFO] Loading sparse MIMIC data ...")
    path = os.path.join(FILE_DIR, "../../data/MIMIC")
    X = sparse.load_npz(os.path.join(path, "mimic_data.npz"))  # CSR matrix
    Y = np.load(os.path.join(path, "label_y.npy"))                 # np array

    print(f"[DEBUG] Raw shape: X={X.shape}, Y={Y.shape}, num_classes={len(np.unique(Y))}")

    nonzero_counts = X.getnnz(axis=1)
    valid_indices = np.where(nonzero_counts >= sparsity_threshold)[0]
    X = X[valid_indices]
    Y = Y[valid_indices]
    print(f"[INFO] After sparsity filtering (≥{sparsity_threshold} non-zero): X={X.shape}, Y={Y.shape}")

    np.random.seed(seed)
    r = np.arange(X.shape[0])
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]

    len_total = X.shape[0]
    if if_avg_split:
        train_classifier_ratio, train_attack_ratio = 0.4, 0.2
    else:
        train_classifier_ratio, train_attack_ratio = float(300) / float(len_total), 0.3

    X_train = X[:int(train_classifier_ratio * len_total)]
    X_attack = X[int(train_classifier_ratio * len_total):int((train_classifier_ratio + train_attack_ratio) * len_total)]
    X_test = X[int((train_classifier_ratio + train_attack_ratio) * len_total):]

    Y_train = Y[:int(train_classifier_ratio * len_total)]
    Y_attack = Y[int(train_classifier_ratio * len_total):int((train_classifier_ratio + train_attack_ratio) * len_total)]
    Y_test = Y[int((train_classifier_ratio + train_attack_ratio) * len_total):]

    def split_half(X_part, Y_part):
        r = np.arange(X_part.shape[0])
        np.random.shuffle(r)
        half = len(r) // 2
        return (X_part[r[:half]], Y_part[r[:half]]), (X_part[r[half:]], Y_part[r[half:]])

    (shadow_train_x, shadow_train_y), (target_train_x, target_train_y) = split_half(X_train, Y_train)
    (shadow_test_x, shadow_test_y), (target_test_x, target_test_y) = split_half(X_test, Y_test)

    def to_tensor_dataset(X_csr, y):
        X_dense = torch.FloatTensor(X_csr.toarray())
        y_tensor = torch.LongTensor(y)
        return torch.utils.data.TensorDataset(X_dense, y_tensor)

    shadow_train = to_tensor_dataset(shadow_train_x, shadow_train_y)
    shadow_test = to_tensor_dataset(shadow_test_x, shadow_test_y)
    target_train = to_tensor_dataset(target_train_x, target_train_y)
    target_test = to_tensor_dataset(target_test_x, target_test_y)
    pseudoattack_data = to_tensor_dataset(X_attack, Y_attack)

    return target_train, target_test, shadow_train, shadow_test, pseudoattack_data
    '''
    
def prepare_mimic(seed=1000, if_avg_split=True):
    """
    MIMIC dataset (mortality binary classification task):
    X: (n_samples, n_features)
    Y: (n_samples,) - binary: 0 (alive), 1 (died)
    """
    print("[INFO] Loading MIMIC data for mortality prediction ...")
    path = os.path.join(FILE_DIR, "../../data/MIMIC")
    data = np.load(os.path.join(path, "mimic_data.npz"))
    X = data["X"]  # (n_samples, n_features)
    Y = data["Y"]  # binary labels: 0=alive, 1=died

    print(f"[DEBUG] Raw shape: X={X.shape}, Y={Y.shape}, Class balance: {np.bincount(Y)}")

    np.random.seed(seed)
    r = np.arange(X.shape[0])
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]

    len_total = X.shape[0]
    train_classifier_ratio, train_attack_ratio = 0.4, 0.2 if if_avg_split else (300.0 / len_total, 0.3)

    X_train = X[:int(train_classifier_ratio * len_total)]
    X_attack = X[int(train_classifier_ratio * len_total):int((train_classifier_ratio + train_attack_ratio) * len_total)]
    X_test = X[int((train_classifier_ratio + train_attack_ratio) * len_total):]

    Y_train = Y[:int(train_classifier_ratio * len_total)]
    Y_attack = Y[int(train_classifier_ratio * len_total):int((train_classifier_ratio + train_attack_ratio) * len_total)]
    Y_test = Y[int((train_classifier_ratio + train_attack_ratio) * len_total):]

    def split_half(X_part, Y_part):
        r = np.arange(X_part.shape[0])
        np.random.shuffle(r)
        half = len(r) // 2
        return (X_part[r[:half]], Y_part[r[:half]]), (X_part[r[half:]], Y_part[r[half:]])

    (shadow_train_x, shadow_train_y), (target_train_x, target_train_y) = split_half(X_train, Y_train)
    (shadow_test_x, shadow_test_y), (target_test_x, target_test_y) = split_half(X_test, Y_test)

    def to_tensor_dataset(X_part, Y_part):
        X_tensor = torch.FloatTensor(X_part)
        Y_tensor = torch.LongTensor(Y_part).view(-1)
        return torch.utils.data.TensorDataset(X_tensor, Y_tensor)

    return (
        to_tensor_dataset(target_train_x, target_train_y),
        to_tensor_dataset(target_test_x, target_test_y),
        to_tensor_dataset(shadow_train_x, shadow_train_y),
        to_tensor_dataset(shadow_test_x, shadow_test_y),
        to_tensor_dataset(X_attack, Y_attack)
    )