import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import init_bn, init_layer

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)    
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
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
        return x
        
class VggishBottleneck(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(VggishBottleneck, self).__init__()
        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=32)
        self.conv_block2 = VggishConvBlock(in_channels=32, out_channels=64)
        self.conv_block3 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block4 = VggishConvBlock(in_channels=128, out_channels=128)

        self.final_conv = nn.Conv2d(in_channels=128, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.final_conv)

    def forward(self, input):
        (_, seq_len, freq_bins) = input.shape

        x = input.view(-1, 1, seq_len, freq_bins)
        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        bottleneck = torch.sigmoid(self.final_conv(x))
        '''(samples_num, classes_num, time_steps, freq_bins)'''
        
        return bottleneck
        
class VggishGMP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(VggishGMP, self).__init__()
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''
        
        # Pool each feature map to a scalar. 
        output = self.global_max_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        output_dict = {'y_pred': output}
        return output_dict
            
    def global_max_pooling(self, input):
        x = F.max_pool2d(input, kernel_size=input.shape[2:])
        output = x.view(x.shape[0], x.shape[1])
        return output       
        
class VggishGAP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(VggishGAP, self).__init__()
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''

        output = self.global_avg_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        output_dict = {'y_pred': output}
        return output_dict
            
    def global_avg_pooling(self, input):
        x = F.avg_pool2d(input, kernel_size=input.shape[2:])
        output = x.view(x.shape[0], x.shape[1])
        return output
        
class VggishGWRP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(VggishGWRP, self).__init__()
        self.seq_len = seq_len
        self.freq_bins = freq_bins
        
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
        decay = 0.9998
        (self.gwrp_w, self.sum_gwrp_w) = self.calculate_gwrp_weights(decay, cuda)
        
    def calculate_gwrp_weights(self, decay, cuda):
        gwrp_w = decay ** np.arange(self.seq_len * self.freq_bins)
        gwrp_w = torch.Tensor(gwrp_w)
        if cuda:
            gwrp_w = gwrp_w.cuda()
            
        sum_gwrp_w = torch.sum(gwrp_w)
        return gwrp_w, sum_gwrp_w
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''
        
        output = self.global_weighted_rank_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        output_dict = {'y_pred': output}
        return output_dict
        
    def global_weighted_rank_pooling(self, input):
        """
        Args:
          input: (batch_size, classes_num, time_steps, freq_bins)
          
        Returns:
          output: (batch_size, classes_num)
        """
        x = input.view((input.shape[0], input.shape[1], 
                        input.shape[2] * input.shape[3]))
        '''(batch_size, classes_num, time_steps * freq_bins)'''
        
        # Sort values in each feature map in descending order. 
        (x, _) = torch.sort(x, dim=-1, descending=True)
        x = x * self.gwrp_w[None, None, :]
        '''(batch_size, classes_num, time_steps * freq_bins)'''
        
        x = torch.sum(x, dim=-1)
        '''(batch_size, classes_num)'''
        output = x / self.sum_gwrp_w
        return output

class Dilation_EmbeddingLayers(nn.Module):
    def __init__(self):
        super(Dilation_EmbeddingLayers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class AttrousCNN_2DAttention(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        super(AttrousCNN_2DAttention, self).__init__()

        self.emb = Dilation_EmbeddingLayers()
        self.attention = Attention2d(256, classes_num, att_activation='sigmoid', cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)

        output = self.attention(x)
        output = torch.sigmoid(output)
        output_dict = {'y_pred': output}
        return output_dict

class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()
        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(in_channels=n_in, out_channels=n_out, \
                            kernel_size=(1, 1), stride=(1, 1), 
                            padding=(0, 0), bias=True)

        self.cla = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), \
                            stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """
        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)
        return x