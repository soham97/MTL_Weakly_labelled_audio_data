import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import init_bn, init_layer
import models_extension

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1,1),
                              padding=(1, 1), bias=False)
                                     
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=1,
                              padding=(1, 1), bias=False)

        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride= 1, padding= 1)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.avgpool(x)

class SEDencoder(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(SEDencoder, self).__init__()
        self.cb1 = Block(in_channels=1, out_channels=32)
        self.cb2 = Block(in_channels=32, out_channels=64)
        self.cb3 = Block(in_channels=64, out_channels=128)
        self.cb4 = Block(in_channels=128, out_channels=256)

        self.class_conv = nn.Conv2d(in_channels=256, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.class_conv)

    def forward(self, input):
        (_, seq_len, freq_bins) = input.shape
        x = input.view(-1, 1, seq_len, freq_bins)

        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)

        class_x = self.class_conv(x)
        '''(samples_num, class_number, time_steps, freq_bins)'''        
        return x, class_x

class AuxillaryDecoder(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(AuxillaryDecoder, self).__init__()
        self.cb1 = Block(in_channels=256, out_channels=128)
        self.cb2 = Block(in_channels=128, out_channels=64)
        self.cb3 = Block(in_channels=64, out_channels=32)
        self.cb4 = Block(in_channels=32, out_channels=1)

    def forward(self, input):
        '''input is of shape (samples_num, feature_maps, time_steps, freq_bins)'''
        x = self.cb1(input)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        '''x is of (samples_num, 1, time_steps, freq_bins)'''

        dec = x.squeeze(1)
        '''dec is of shape (samples_num, time_steps, freq_bins)'''
        return dec

class DualStageAttention(nn.Module):
    def __init__(self, seq_len, freq_bins):
        self.fc_mel_prob = nn.Linear(freq_bins, freq_bins)
        self.fc_mel_att = nn.Linear(freq_bins, freq_bins)  

        self.fc_time_prob = nn.Linear(seq_len, seq_len) # operates on (bs, time, class)
        self.fc_time_att = nn.Linear(seq_len, seq_len) # operates on (bs, time, class)
    
    def forward(self, class_x):
        """ 
        Args:
          class_x: (batch_size, classes_num, time_steps, freq_bins)
          
        Returns:
          output_dict: dictionary containing y_pred and attention weights
        """
        mel_probs = torch.sigmoid(self.fc_mel_prob(class_x))
        mel_attw = F.softmax(self.fc_mel_att(class_x), dim = -1)
        mel_x = (mel_probs * mel_attw).sum(dim = -1)
        mel_x = mel_x.squeeze(-1)
        '''
            mel_probs: (batch_size, classes_num, time_steps, freq_bins)
            mel_attw: (batch_size, classes_num, time_steps, freq_bins)
            mel_x: (batch_size, classes_num, time_steps)
        '''
        time_probs = torch.sigmoid(self.fc_time_prob(mel_x))
        time_attw = F.softmax(self.fc_time_att(mel_x), dim = -1)
        time_x = (time_probs * time_attw).sum(dim = -1)
        time_x = time_x.clamp(0, 1)
        out = time_x.squeeze(-1)
        '''
            time_probs: (batch_size, classes_num, time_steps)
            time_attw: (batch_size, classes_num, time_steps)
            time_x: (batch_size, classes_num)
        '''
        return mel_attw, time_attw, mel_x, time_x, out

class MTL_SEDNetwork(nn.Module):  
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(MTL_SEDNetwork, self).__init__()
        self.enc = SEDencoder(classes_num, seq_len, freq_bins, cuda)
        self.dec = AuxillaryDecoder(classes_num, seq_len, freq_bins, cuda)
        self.dsa = DualStageAttention(seq_len, freq_bins)

    def forward(self, input):
        """ 
        Args:
          input: (batch_size, time_steps, freq_bins)
          
        Returns:
          output_dict: dictionary containing y_pred and attention weights
        """
        x, class_x = self.enc(input)
        '''
            x: (batch_size, filters, time_steps, freq_bins)
            class_x: (batch_size, classes_num, time_steps, freq_bins)
        '''
        input_x = self.dec(x)
        '''
            input_x: (batch_size, time_steps, freq_bins)
        '''
        mel_attw, time_attw, mel_x, time_x, out = self.dsa(class_x)
        output_dict = {'y_pred': out, 'x_rec': input_x, 'class_wise_input': class_x, \
                        'mel_attw':mel_attw, 'time_attw':time_attw, 'mel_x': mel_x, 'time_x':time_x}
        return output_dict

def get_model(model_type):
    if model_type == 'MTL_SEDNetwork': 
        return MTL_SEDNetwork
    if model_type =='GMP':
        return models_extension.VggishGMP
    if model_type =='GAP':
        return models_extension.VggishGAP
    if model_type =='GWRP':
        return models_extension.VggishGWRP
    if model_type =='AttrousCNN_2DAttention':
        return models_extension.AttrousCNN_2DAttention
    else:
        raise Exception('Incorrect model type!')
