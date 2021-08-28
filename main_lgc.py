import logging
import os
import random
import time
import argparse
import configparser
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from script import dataloader, utility, earlystopping
from model import models

import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is so important for multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='LGC')
    parser.add_argument('--mode', type=str, default='test', help='running mode, \
                        default as test, tuning as alternative')
    parser.add_argument('--enable_cuda', type=bool, default=True, \
                        help='enable or disable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=100, help='the random seed')
    parser.add_argument('--dataset_config_path', type=str, default='./config/data/cora.ini', \
                        help='the path of dataset config file, cora.ini for CoRA')
    parser.add_argument('--model_config_path', type=str, default='./config/model/lgc_sym.ini', \
                        help='the path of model config file')
    parser.add_argument('--gamma', type=float, default=2, help='gamma >= 2, defaut as 2')
    parser.add_argument('--enable_bias', type=bool, default=True, help='enable to use bias in graph convolution layers or not')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer, default as Adam')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='early stopping patience, default as 50')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    if args.mode != 'test' and args.mode != 'tuning':
        raise ValueError(f'ERROR: Wrong running mode')
    else:
        mode = args.mode

    SEED = args.seed
    set_env(SEED)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is so important for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    config = configparser.ConfigParser()
    def ConfigSectionMap(section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
                if dict1[option] == -1:
                    logging.debug('skip: %s' % option)
            except:
                print('exception on %s!' % option)
                dict1[option] = None
        return dict1

    dataset_config_path = args.dataset_config_path
    model_config_path = args.model_config_path

    config.read(dataset_config_path, encoding='utf-8')
    dataset_name = ConfigSectionMap('dataset')['name']
    data_path = './data/'
    learning_rate = float(ConfigSectionMap('model')['learning_rate'])
    weight_decay_rate = float(ConfigSectionMap('model')['weight_decay_rate'])
    model_save_path = ConfigSectionMap('model')['model_save_path']

    config.read(model_config_path, encoding='utf-8')
    model_name = ConfigSectionMap('model')['name']
    norm_lap_type = ConfigSectionMap('gconv')['norm_lap_type']
    if norm_lap_type != 'sym' and norm_lap_type != 'rw':
        raise ValueError(f'ERROR: The type of normalized Laplacian matrix {norm_lap_type} is undefined.')

    if mode == 'tuning':
        param = nni.get_next_parameter()
        gamma = [*param.values()][0]
    else:
        if args.gamma < 2:
            raise ValueError(f'ERROR: The value of gamma is unacceptable.')
        else:
            gamma = args.gamma
    
    enable_bias = args.enable_bias
    epochs = args.epochs
    opt = args.opt
    early_stopping_patience = args.early_stopping_patience

    model_save_path = model_save_path + model_name + '_' + norm_lap_type + '_' + str(gamma) + '_gamma_' + '.pth'

    return device, dataset_name, data_path, learning_rate, weight_decay_rate, model_name, \
            model_save_path, gamma, norm_lap_type, enable_bias, epochs, opt, early_stopping_patience
    
def process_data(device, dataset_name, data_path, norm_lap_type, gamma):

    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed':
        features, dir_adj, g, labels, idx_train, idx_val, idx_test = dataloader.load_citation_data(dataset_name, data_path)
    elif dataset_name == 'cornell' or dataset_name == 'texas' or dataset_name == 'washington' or dataset_name == 'wisconsin':
        features, dir_adj, g, labels, idx_train, idx_val, idx_test = dataloader.load_webkb_data(dataset_name, data_path)

    n_vertex, n_feat, n_labels, n_class = features.shape[0], features.shape[1], labels.shape[0], labels.shape[1]
    labels = np.argmax(labels, axis=1)

    if norm_lap_type == 'sym':
        norm_lap = utility.calc_sym_lap(dir_adj)
    elif norm_lap_type == 'rw':
        norm_lap = utility.calc_rw_lap(dir_adj)

    features = utility.calc_lgc_features(norm_lap, features, gamma)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # from matrix to tensor
    # move tensor to device
    features = torch.from_numpy(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    return features, labels, idx_train, idx_val, idx_test, n_feat, n_class, n_vertex

def prepare_model(n_feat, n_class, n_vertex, enable_bias, early_stopping_patience, learning_rate, \
                weight_decay_rate, model_save_path, opt):
    model = models.LGC(n_feat, n_class, enable_bias).to(device)

    loss = nn.NLLLoss()
    early_stopping = earlystopping.EarlyStopping(patience=early_stopping_patience, path=model_save_path, verbose=True)

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return model, loss, early_stopping, optimizer, scheduler

def train(epochs, model, optimizer, scheduler, early_stopping, features, labels, loss, idx_train, idx_val):
    train_time_list = []
    for epoch in range(epochs):
        train_epoch_begin_time = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = loss(output[idx_train], labels[idx_train])
        acc_train = utility.calc_accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        #scheduler.step()
        train_epoch_end_time = time.time()
        train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
        train_time_list.append(train_epoch_time_duration)

        loss_val, acc_val = val(model, labels, output, loss, idx_val)
        print('Epoch: {:03d} | Learning rate: {:.8f} | Train loss: {:.6f} | Train acc: {:.6f} | Val loss: {:.6f} | Val acc: {:.6f} | Training duration: {:.6f}'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), train_epoch_time_duration))
        nni.report_intermediate_result(acc_val.item())

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print('Early stopping.')
            break
    
    mean_train_epoch_time_duration = np.mean(train_time_list)
    print('\nTraining finished.\n')

    return mean_train_epoch_time_duration

def val(model, labels, output, loss, idx_val):
    model.eval()
    with torch.no_grad():
        loss_val = loss(output[idx_val], labels[idx_val])
        acc_val = utility.calc_accuracy(output[idx_val], labels[idx_val])
    
    return loss_val, acc_val

def test(model, model_save_path, features, labels, loss, idx_test, model_name, dataset_name, mean_train_epoch_time_duration):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_test = loss(output[idx_test], labels[idx_test])
        acc_test = utility.calc_accuracy(output[idx_test], labels[idx_test])
        print('Model: {} | Dataset: {} | Test loss: {:.6f} | Test acc: {:.6f} | Mean training duration for each epoch: {:.6f}'.\
            format(model_name, dataset_name, loss_test.item(), acc_test.item(), mean_train_epoch_time_duration))
        nni.report_final_result(acc_test.item())

if __name__ == "__main__":
    device, dataset_name, data_path, learning_rate, weight_decay_rate, model_name, model_save_path, gamma, norm_lap_type, enable_bias, epochs, opt, early_stopping_patience = get_parameters()

    features, labels, idx_train, idx_val, idx_test, n_feat, n_class, n_vertex = process_data(device, dataset_name, data_path, norm_lap_type, gamma)

    model, loss, early_stopping, optimizer, scheduler = prepare_model(n_feat, n_class, n_vertex, enable_bias, early_stopping_patience, learning_rate, weight_decay_rate, model_save_path, opt)

    mean_train_epoch_time_duration = train(epochs, model, optimizer, scheduler, early_stopping, features, labels, loss, idx_train, idx_val)
    test(model, model_save_path, features, labels, loss, idx_test, model_name, dataset_name, mean_train_epoch_time_duration)