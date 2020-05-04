import os
import numpy as np
import logging
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import config

def create_folder(fd):
    if not os.path.exists(fd):
        # creates problems when multiple scripts creating folders
        try:
            os.makedirs(fd)
        except:
            print('Folder already exits')

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging

def save_model(e_, model, optimizer, scheduler, path_model = 'best.pth'):
    checkpoint = { 
        'epoch': e_,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, path_model)

def load_model(path_model, cuda):
    return torch.load(path_model) if cuda else torch.load(path_model, map_location=torch.device('cpu'))

def get_stats(y_true_train, y_pred_train, y_true_val, y_pred_val):
    d = {}
    y_true_train = np.concatenate(y_true_train, axis = 0)
    y_pred_train = np.concatenate(y_pred_train, axis = 0)
    y_true_val = np.concatenate(y_true_val, axis = 0)
    y_pred_val = np.concatenate(y_pred_val, axis = 0)
    
    d['roc_auc_train'] = roc_auc_score(y_true_train, y_pred_train)
    d['ap_macro_train'] = average_precision_score(y_true_train, y_pred_train, average = 'macro')
    d['ap_micro_train'] = average_precision_score(y_true_train, y_pred_train, average = 'micro')

    d['roc_auc_val'] = roc_auc_score(y_true_val, y_pred_val)
    d['ap_macro_val'] = average_precision_score(y_true_val, y_pred_val, average = 'macro')
    d['ap_micro_val'] = average_precision_score(y_true_val, y_pred_val, average = 'micro')
    return d

def reconstruction_plot(x, x_rec, args, i, path_results):
    """
        x: torch.Size([1, time_steps, freq_bins])
        x_rec:  torch.Size([1, time_steps, freq_bins])
    """
    path_png = '{}.png'.format(i)

    # set plot visual settings
    fig, axs = plt.subplots(2, 1, figsize = (10,7))
    fig.tight_layout(pad=3.0)
    small_font = 8
    medium_font = 10
    big_font = 12
    plt.rc('font', size=small_font)          # controls default text sizes
    plt.rc('axes', titlesize=medium_font)    # fontsize of the axes title
    plt.rc('axes', labelsize=medium_font)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_font)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_font)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_font)    # legend fontsize
    plt.rc('figure', titlesize=big_font)     # fontsize of the figure title

    # add the required plots
    axs[0].matshow(x.T, origin='lower', aspect='auto', cmap='jet')
    axs[0].xaxis.set_ticks(np.arange(0,310, 309))
    axs[0].set_title('Scaled Input Mel Spectrogram')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mel bins')

    axs[1].matshow(x_rec.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].xaxis.set_ticks(np.arange(0,310, 309))
    axs[1].set_title('Scaled Output Mel Spectrogram')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('mel bins')

    plt.savefig(os.path.join(path_results, path_png))
    plt.close('all')

def attention_plot(mel_x, mel_attw, time_x, time_attw, args, i, path_results):
    """
        x: torch.Size([1, time_steps, freq_bins])
        x_rec:  torch.Size([1, time_steps, freq_bins])
    """
    path_png = '{}.png'.format(i)

    # set plot visual settings
    fig, axs = plt.subplots(5, 1, figsize = (12,14))
    fig.tight_layout(pad=3.0)
    small_font = 8
    medium_font = 10
    big_font = 12
    plt.rc('font', size=small_font)          # controls default text sizes
    plt.rc('axes', titlesize=medium_font)    # fontsize of the axes title
    plt.rc('axes', labelsize=medium_font)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_font)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_font)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_font)    # legend fontsize
    plt.rc('figure', titlesize=big_font)     # fontsize of the figure title

    top_class_ids = time_x.argsort()[-3:][::-1]
    labels = [config.ix_to_lb[i] for i in top_class_ids]
    axs[0].matshow(mel_attw[top_class_ids[0]].T, origin='lower', aspect='auto', cmap='jet')
    axs[0].xaxis.set_ticks(np.arange(0,310, 309))
    axs[0].set_title('Mel bin attention weights for class: ' + str(labels[0]))
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('classes')

    axs[1].matshow(mel_attw[top_class_ids[1]].T, origin='lower', aspect='auto', cmap='jet')
    axs[1].xaxis.set_ticks(np.arange(0,310, 309))
    axs[1].set_title('Mel bin attention weights for class: ' + str(labels[1]))
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('classes')

    axs[2].matshow(mel_x, origin='lower', aspect='auto', cmap='jet')
    axs[2].xaxis.set_ticks(np.arange(0,310, 309))
    axs[2].set_title('Ouput of Mel bin attention')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('classes')

    axs[3].matshow(time_attw, origin='lower', aspect='auto', cmap='jet')
    axs[3].xaxis.set_ticks(np.arange(0,310, 309))
    axs[3].set_title('Time attention weights')
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('classes')

    axs[4].plot(time_x)
    axs[4].xaxis.set_ticks(np.arange(0,41))
    axs[4].set_title('Output of time attention')
    axs[4].set_xlabel('class number')
    axs[4].set_ylabel('probability')

    plt.savefig(os.path.join(path_results, path_png))
    plt.close('all')



