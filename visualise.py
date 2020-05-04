import os
import yaml
import time
import argparse
import torch.nn as nn
import torch
import logging
import config
from utils import create_folder, create_logging, get_stats, load_model
from train_helper import get_dataloader, visualise_epoch
from models import get_model

if __name__ == '__main__':
    """
    Example usage:

    python3 main.py -data_dir data -exp_name dual_attn_vis -batch_size 24 \
        -num_workers 64 -data_parallel 1 -model_type MTL_SEDNetwork -snr 20 \
        -pretrained_model_path test/best.pth -val_fold 4
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", type=str, default="data")
    parser.add_argument("-exp_name", "--exp_name", type=str, help="Name of experiment")
    parser.add_argument("-model_type", "--model_type", type = str, choices=["MTL_SEDNetwork","GMP", "GAP", "GWRP", "AttrousCNN_2DAttention"])
    parser.add_argument("-batch_size", "--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=64, help = "Number of workers to be used")
    parser.add_argument("-data_parallel", "--data_parallel", type = int, help="1 if model is to be distributed across multiple GPUs")
    parser.add_argument("-snr", "--snr", required=True, type=int, help="SNR between 0, 10, 20")
    parser.add_argument("-val_fold", "--val_fold", required=True, type=int, help="holdout fold between 1,2,3,4")
    parser.add_argument("-pretrained_model_path", "--pretrained_model_path", required = True, type=str, help="path of model to use for generating visualations")
    args = parser.parse_args()

    # Setting up experiments
    data_parallel = args.data_parallel == 1
    if args.model_type != "MTL_SEDNetwork":
        print('Only support visualisation for MTL_SEDNetwork')
        exit()
    cuda = torch.cuda.is_available() 
    create_folder('visulations')
    base_path = os.path.join('visulations', args.exp_name)
    create_logging(base_path, filemode = 'w')
    logging.info(f'logging started for visualisation experiment = {args.exp_name}')

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
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)
    if cuda:
        model = model.cuda()

    checkpoint = load_model(args.pretrained_model_path, cuda)
    model.load_state_dict(checkpoint['model'])

    # generate plots
    visualise_epoch(data_container, model, args, cuda, base_path)






