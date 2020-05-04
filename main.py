import os
import yaml
import time
import argparse
import torch.nn as nn
import torch
import logging
import config
from utils import create_folder, create_logging, get_stats, save_model
from train_helper import get_dataloader, train_epoch, eval_epoch
from models import get_model

if __name__ == '__main__':
    """
    Example usage:

    python3 main.py -data_dir data -exp_name test -batch_size 24 -epochs 80 -lr 1e-3 \
        -num_workers 64 -data_parallel 1 -model_type MTL_SEDNetwork -val_fold 4 -snr 20 -alpha 0.001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", type=str, default="data")
    parser.add_argument("-exp_name", "--exp_name", type=str, help="Name of experiment")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("-epochs", "--epochs", type=int, default=50, help="Number of epochs to train model")
    parser.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning rate for optimiser")
    parser.add_argument("-w_decay", "--w_decay", type=float, default=0.0)
    parser.add_argument("-num_workers", "--num_workers", type=int, default=64, help = "Number of workers to be used")
    parser.add_argument("-data_parallel", "--data_parallel", type = int, help="1 if model is to be distributed across multiple GPUs")
    parser.add_argument("-model_type", "--model_type", type = str, choices=["MTL_SEDNetwork","GMP", "GAP", "GWRP", "AttrousCNN_2DAttention"])
    parser.add_argument("-val_fold", "--val_fold", required=True, type=int, help="holdout fold between 1,2,3,4")
    parser.add_argument("-snr", "--snr", required=True, type=int, help="SNR between 0, 10, 20")
    parser.add_argument("-alpha", "--alpha", required=True, type=float, help="Alpha (weightage) for reconstruction loss")
    args = parser.parse_args()

    # Setting up experiments
    data_parallel = args.data_parallel == 1
    MTL = args.model_type == "MTL_SEDNetwork" # if true use auxillary task MTL loss
    cuda = torch.cuda.is_available() 
    create_folder('experiments')
    base_path = os.path.join('experiments', args.exp_name)
    model_path = os.path.join(base_path, 'best.pth')
    create_folder(base_path)
    create_logging(base_path, filemode = 'w')
    logging.info(f'logging started for experiment = {args.exp_name} and validation fold = {args.val_fold}')

    # Data and yaml path
    data_path = os.path.join(args.data_dir, f'logmel/logmel_snr_{args.snr}.h5')
    yaml_path = os.path.join(args.data_dir, 'mixture.yaml')
    with open(yaml_path, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    # Dataset and Dataloader
    data_container = get_dataloader(data_path, yaml_path, args, cuda)

    # model and dataparallel
    Model = get_model(args.model_type)
    model = Model(config.classes_num, config.seq_len, config.mel_bins, cuda)
    if cuda and torch.cuda.device_count() > 1 and data_parallel:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)
    if cuda:
        model = model.cuda()

    # Loss functions, optimsier, scheduler
    criterian_SED = nn.BCELoss(reduction = 'mean') # SED -> Sound event detection 
    criterian_AT = nn.MSELoss(reduction = 'mean') # AT -> Auxillary task
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 3, verbose = True)

    logging.info('-------------------------------------------------------')
    best_micro_val = 0.0
    for e in range(args.epochs):
        s = time.time()

        train_loss, y_pred_train, y_true_train = train_epoch(data_container, model, optimizer, criterian_SED, criterian_AT , scheduler, args, cuda, MTL)
        val_loss, y_pred_val, y_true_val = eval_epoch(data_container, model, optimizer, criterian_SED, criterian_AT , scheduler, args, cuda, MTL)
        d = get_stats(y_true_train, y_pred_train, y_true_val, y_pred_val)
        scheduler.step(d['ap_micro_val'])

        if d['ap_micro_val'] > best_micro_val:
            best_micro_val = d['ap_micro_val']
            save_model(e, model, optimizer, scheduler, model_path)
        
        # ap_micro is of primary interest here
        logging.info('[EPOCH] {}'.format(e))
        logging.info('[TRAIN] loss: {:.3f}, roc_auc: {:.3f}, ap_macro: {:.3f}, ap_micro: {:.3f}'.format(train_loss, d['roc_auc_train'], d['ap_macro_train'], d['ap_micro_train']))
        logging.info('[VAL] loss: {:.3f}, roc_auc: {:.3f}, ap_macro: {:.3f}, ap_micro: {:.3f}'.format(val_loss, d['roc_auc_val'], d['ap_macro_val'], d['ap_micro_val']))
        logging.info('[TIME] {:.3f} sec'.format(time.time() - s))
        logging.info('-------------------------------------------------------')