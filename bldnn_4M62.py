"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_custom.torch_utils import to_gpu
from torch_custom.stft_helper import StftHelper
import torch_custom.spectral_ops as spo

from torch_custom.custom_layers import CustomModel, update_dict
from torch_custom.custom_layers import Dense
from torch_custom.custom_layers import GlobalMeanVarianceNormalizer as GMVN


class LSTMLinear(nn.Module):
  def __init__(self, in_dim, hid_dim, proj_dim=0, num_layers=1, bias=True, 
               batch_first=True, dropout=0.0, bidirectional=True, 
               scope='LSTMLinear'):
    super(LSTMLinear, self).__init__()
    self.weights = {}

    self.lstm = nn.LSTM(
      in_dim, hid_dim, num_layers, bias=bias, 
      batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    for name, param in self.lstm._parameters.items():
      if name.startswith('weight'):
        # self.weights.append(param)
        self.weights[scope+'_'+name] = param
        # print(name, param.shape)

    num_directions = 2 if bidirectional else 1
    self.batch_dim = 0 if batch_first else 1
    self.hid_dim, self.proj_dim = hid_dim, proj_dim
    self.tot_layers = num_layers * num_directions

    if proj_dim:
      self.projector = nn.Linear(
        hid_dim*num_directions, proj_dim, bias=False)
      # self.weights.append(self.projector.weight)
      self.weights[scope+'_projection'] = self.projector.weight

  def forward(self, x, drop=0.0):
    """ x in (T, B, C) (batch_first=False) or (B, T, C) (batch_first=True) """
    batch_size = x.size(self.batch_dim)

    ## Forward Op
    h0 = torch.zeros(self.tot_layers, batch_size, self.hid_dim).to(x.device)
    c0 = torch.zeros(self.tot_layers, batch_size, self.hid_dim).to(x.device)
    # x, _ = self.lstm(x)  # automatically set initial states to zeros
    # print(h0.shape, c0.shape)
    # print('x', x.shape)
    x, _ = self.lstm(x, (h0, c0))
    # print('x', x.shape)
    if self.proj_dim:
      x = self.projector(x)
      # print('x_proj', x.shape)
    if drop:
      x = F.dropout(x, drop, training=self.training)
    return x


class LstmDnnNet(CustomModel):
  def __init__(self, input_dim=513, hidden_dim=400, num_layers=1, bias=True, 
               lstm_dropout=0.0, bidirectional=True, dense_dim=800, 
               stft_opts={}, scope='LstmDnnNet', 
               input_norm='globalmvn', mean_in=None, std_in=None, 
               mean_out=None, std_out=None):
    super(LstmDnnNet, self).__init__()
    self.weights = {}

    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)
      # n_fft=1024, hop_length=256, win_length=1024, win_type='hanning', 
      # symmetric=True)#, pad_both=True, snip_edges=False)

    ## LSTM layers
    self.lstm = LSTMLinear(
      input_dim, hidden_dim, proj_dim=0, num_layers=num_layers, bias=bias, 
      batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional, 
      scope='LSTMLinear')

    ## Dense layers
    self.activ_func = nn.ELU(inplace=True)
    self.dense1 = Dense(2*hidden_dim, dense_dim, bias=bias, scope='dense1')
    self.dense2 = Dense(dense_dim, dense_dim, bias=bias, scope='dense2')
    self.dense_out = Dense(dense_dim, input_dim, bias=bias, scope='dense_out')

    ## Kernels subject to l2-regularization
    update_dict(self.weights, self.lstm.weights)
    update_dict(self.weights, self.dense1.weights)
    update_dict(self.weights, self.dense2.weights)
    update_dict(self.weights, self.dense_out.weights)
    self.weights = {scope+'/'+k:v for k,v in self.weights.items()}
    # self.weights_list = list(self.weights.values())
    self.weights_name = list(self.weights.keys())
    self.weights_list = [self.weights[name] for name in self.weights_name]

    ## Input normalization method
    if input_norm == 'batchnorm':
      self.input_norm = nn.BatchNorm1d(input_dim)
    elif input_norm =='globalmvn':
      if mean_in is None:
        mean_in = np.zeros((1, input_dim, 1), dtype='float32')
      if std_in is None:
        std_in = np.ones((1, input_dim, 1), dtype='float32')
      self.input_norm = GMVN(mean_in, std_in)

    ## Output normalization method
    if mean_out is None:
      mean_out = np.zeros((1, input_dim, 1), dtype='float32')
    if std_out is None:
      std_out = np.ones((1, input_dim, 1), dtype='float32')
    self.output_norm = GMVN(mean_out, std_out)

    ## Loss functions for training this model
    self.loss_mse = nn.MSELoss(reduction='mean')
    # self.loss_mae = nn.L1Loss(reduction='mean')

  @staticmethod
  def parse_batch(data, target, device):
    data, target = to_gpu(data, device), to_gpu(target, device)
    return data, target

  def forward(self, x, drop=0.0):
    ## If the input is batched waveforms, compute LPS features;
    ## otherwise, the input must be LPS features.
    if x.dim() < 3:
      x = self.stft_helper.stft(x) # (B,F,T,2)
      x = spo.stft2lps(x)          # (B,F,T)

    ## Input normalization
    x = self.input_norm(x)

    ## LSTM layers
    x = x.transpose(1, 2) # (B,T,F)
    x = self.lstm(x, drop)

    ## Dense layers
    x = self.activ_func(self.dense1(x))
    if drop:
      x = F.dropout(x, drop, training=self.training)
    x = self.activ_func(self.dense2(x))
    if drop:
      x = F.dropout(x, drop, training=self.training)
    x = self.dense_out(x)

    ## Output normalization
    x = x.transpose(1, 2)
    x = self.output_norm.denormalize(x)
    return x

  def get_loss(self, data, target, drop=0.0, summarize=False):
    """ Both "data" and "target" are batched time-domain waveforms """
    output_lps = self.forward(data, drop=drop)
    target_lps = spo.stft2lps(self.stft_helper.stft(target))
    raw_loss = self.loss_mse(output_lps, target_lps)
    if not summarize:
      return raw_loss
    else:
      loss_dict = {"raw_loss":raw_loss.item()}
      return raw_loss, loss_dict, (output_lps[-1], target_lps[-1])

