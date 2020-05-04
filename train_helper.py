import os
from dataset import audio_dataset
from torch.utils.data import DataLoader
from utils import reconstruction_plot, attention_plot, create_folder

def get_dataloader(data_path, yaml_path, args, cuda):
    train_dataset = audio_dataset(data_path, yaml_path, args.val_fold, train = True)
    val_dataset = audio_dataset(data_path, yaml_path, args.val_fold, train = False)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = cuda)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle = False, num_workers= args.num_workers, pin_memory = cuda)

    data_container = {'train_dataset': train_dataset, 'val_dataset': val_dataset, \
                    'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader}
    return data_container

def train_epoch(data_container, model, optimizer, criterian_SED, criterian_AT , scheduler, args, cuda, MTL):
    train_loss = 0.0
    y_pred_train = []
    y_true_train = []
    model.train()
    for x, y, _ in data_container['train_dataloader']:
        if cuda:
            x = x.cuda()
            y = y.cuda()
    
        optimizer.zero_grad()
        out_dict = model(x)
        y_pred = out_dict['y_pred'] 
        loss = criterian_SED(y_pred, y) + (args.alpha*criterian_AT(out_dict['x_rec'], x) if MTL else 0.0)

        loss.backward()
        optimizer.step()
        train_loss += loss.data/(data_container['train_dataset'].__len__()/args.batch_size)
        y_pred_train.append(y_pred.detach().cpu().numpy())
        y_true_train.append(y.detach().cpu().numpy())
    
    return train_loss, y_pred_train, y_true_train

def eval_epoch(data_container, model, optimizer, criterian_SED, criterian_AT , scheduler, args, cuda, MTL):
    val_loss = 0.0
    y_pred_val = []
    y_true_val = []
    model.eval()
    for x, y, _ in data_container['val_dataloader']:
        if cuda:
            x = x.cuda()
            y = y.cuda()
    
        out_dict = model(x)
        y_pred = out_dict['y_pred'] 
        loss = criterian_SED(y_pred, y) + (args.alpha*criterian_AT(out_dict['x_rec'], x) if MTL else 0.0)

        val_loss += loss.data/(data_container['val_dataset'].__len__()/args.batch_size)
        y_pred_val.append(y_pred.detach().cpu().numpy())
        y_true_val.append(y.detach().cpu().numpy())
    
    return val_loss, y_pred_val, y_true_val

def visualise_epoch(data_container, model, args, cuda, base_path):
    base_path_ae = os.path.join(base_path, 'ae_vis')
    base_path_dualatt = os.path.join(base_path, 'dualatt_vis')
    create_folder(base_path_ae)
    create_folder(base_path_dualatt)

    model.eval()
    i = 0
    for x, _, audio_names in data_container['val_dataloader']:
        if cuda:
            x = x.cuda()
        out_dict = model(x)
        y_pred = out_dict['y_pred'].cpu().detach().numpy()
        x_rec = out_dict['x_rec'].cpu().detach().numpy()
        class_x = out_dict['class_wise_input'].cpu().detach().numpy()
        mel_attw = out_dict['mel_attw'].cpu().detach().numpy()
        time_attw = out_dict['time_attw'].cpu().detach().numpy()
        mel_x = out_dict['mel_x'].cpu().detach().numpy()
        time_x = out_dict['time_x'].cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        # here i maintains sample count (global)
        # here j maintains count inside batch (local)
        for j in range(x.shape[0]):
            reconstruction_plot(x[j], x_rec[j], args, audio_names[j], base_path_ae)
            attention_plot(mel_x[j], mel_attw[j], time_x[j], time_attw[j], args, audio_names[j], base_path_dualatt)
            i = i + 1