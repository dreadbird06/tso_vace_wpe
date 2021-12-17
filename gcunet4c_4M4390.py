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
from torch_custom.custom_layers import Glu2d, Conv2d, Conv2dTp
from torch_custom.custom_layers import GlobalMeanVarianceNormalizer as GMVN


class GCUNet3Encoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='GCUNet3Encoder'):
    super(GCUNet3Encoder, self).__init__()
    self.weights = {}

    self.glu2d_1 = Glu2d(
      in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_1')

    self.glu2d_2 = Glu2d(
      out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_2')

    self.conv2d_3 = Conv2d(
      out_channels, out_channels, kernel_size, stride=stride, padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/conv2d_3')

    update_dict(self.weights, self.glu2d_1.weights)
    update_dict(self.weights, self.glu2d_2.weights)
    update_dict(self.weights, self.conv2d_3.weights)

  def forward(self, x, drop=0.0):
    # print('x', x.size())
    h2 = self.glu2d_2(self.glu2d_1(x, drop=drop))
    # print('h2', h2.size())
    hd = self.conv2d_3(h2, drop=drop)
    # print('hd', hd.size())
    return hd, h2

class GCUNet3Decoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='GCUNet3Decoder', out_padding=0, aux_dim=0):
    super(GCUNet3Decoder, self).__init__()
    assert isinstance(out_channels, list) and len(out_channels) == 2
    self.weights = {}

    self.conv2dtp_3 = Conv2dTp(
      in_channels, out_channels[0], kernel_size, stride=stride, padding=padding, 
      out_padding=out_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, 
      scope=scope+'/conv2dtp_3')

    self.glu2d_2 = Glu2d(
      out_channels[0]+aux_dim, out_channels[0], kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_2')

    self.glu2d_1 = Glu2d(
      out_channels[0], out_channels[1], kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_1')

    update_dict(self.weights, self.conv2dtp_3.weights)
    update_dict(self.weights, self.glu2d_2.weights)
    update_dict(self.weights, self.glu2d_1.weights)

  def forward(self, input, output_size=None, input_aux=None, drop=0.0):
    # print('input', input.size())
    hu = self.conv2dtp_3(input, output_size=output_size)
    # print('hu', hu.size())
    if input_aux is not None:
      hu = torch.cat((hu, input_aux), dim=1)
      # print('hu_cat', hu.size())
    if drop:
      hu = F.dropout(hu, p=drop, training=self.training, inplace=True)
    h1 = self.glu2d_1(self.glu2d_2(hu, drop=drop))
    # print('h1', h1.size())
    return h1


class GCUNet2Encoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='GCUNet2Encoder'):
    super(GCUNet2Encoder, self).__init__()
    self.weights = {}

    self.glu2d_1 = Glu2d(
      in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_1')

    self.conv2d_2 = Conv2d(
      out_channels, out_channels, kernel_size, stride=stride, padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/conv2d_2')

    update_dict(self.weights, self.glu2d_1.weights)
    update_dict(self.weights, self.conv2d_2.weights)

  def forward(self, x, drop=0.0):
    # print('x', x.size())
    h1 = self.glu2d_1(x)
    # print('h1', h1.size())
    hd = self.conv2d_2(h1, drop=drop)
    # print('hd', hd.size())
    return hd, h1

class GCUNet2Decoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='GCUNet2Decoder', out_padding=0, aux_dim=0):
    super(GCUNet2Decoder, self).__init__()
    assert isinstance(out_channels, list) and len(out_channels) == 2
    self.weights = {}

    self.conv2dtp_2 = Conv2dTp(
      in_channels, out_channels[0], kernel_size, stride=stride, padding=padding, 
      out_padding=out_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, 
      scope=scope+'/conv2dtp_2')

    self.glu2d_1 = Glu2d(
      out_channels[0]+aux_dim, out_channels[1], kernel_size, stride=(1, 1), padding=padding, 
      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
      scope=scope+'/glu2d_1')

    update_dict(self.weights, self.conv2dtp_2.weights)
    update_dict(self.weights, self.glu2d_1.weights)

  def forward(self, input, output_size=None, input_aux=None, drop=0.0):
    # print('input', input.size())
    hu = self.conv2dtp_2(input, output_size=output_size)
    # print('hu', hu.size())
    if input_aux is not None:
      hu = torch.cat((hu, input_aux), dim=1)
      # print('hu_cat', hu.size())
    if drop:
      hu = F.dropout(hu, p=drop, training=self.training, inplace=True)
    h1 = self.glu2d_1(hu)
    # print('h1', h1.size())
    return h1


class VACENet(CustomModel):
  def __init__(self, input_dim=513, stft_opts={}, fake_scale=2.0, 
               scope='VACENet', input_norm='globalmvn', 
               mean_in=None, std_in=None, mean_out=None, std_out=None):
    super(VACENet, self).__init__()
    self.weights = {}

    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)
    self.fake_scale = fake_scale

    ## First encoder stream
    self.renc1 = GCUNet3Encoder(
      2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='renc1')

    self.renc2 = GCUNet3Encoder(
      32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='renc2')

    self.renc3 = GCUNet2Encoder(
      64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='renc3')

    self.renc4 = GCUNet2Encoder(
      96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='renc4')

    ## Second encoder stream
    self.ienc1 = GCUNet3Encoder(
      2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='ienc1')

    self.ienc2 = GCUNet3Encoder(
      32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='ienc2')

    self.ienc3 = GCUNet2Encoder(
      64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='ienc3')

    self.ienc4 = GCUNet2Encoder(
      96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='ienc4')


    ## Bottleneck Conv2D layers
    self.rconv_mid = Conv2d(
      128, 128, (3, 3), (1, 1), padding=(1, 1), bias=True, 
      scope='rconv_mid')

    self.iconv_mid = Conv2d(
      128, 128, (3, 3), (1, 1), padding=(1, 1), bias=True, 
      scope='iconv_mid')


    ## Decoder stream
    du4, du3, du2, du1 = (256, 128, 64, 32)
    self.du4, self.du3, self.du2, self.du1 = du4, du3, du2, du1

    self.dec4 = GCUNet2Decoder(
      256, [du4, 128], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='dec4', aux_dim=128*2)

    self.dec3 = GCUNet2Decoder(
      128, [du3, 64], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='dec3', aux_dim=96*2)

    self.dec2 = GCUNet3Decoder(
      64, [du2, 32], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='dec2', aux_dim=64*2)

    self.dec1 = GCUNet3Decoder(
      32, [du1, 32], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
      dilation=1, groups=1, bias=True, padding_mode='zeros', 
      scope='dec1', aux_dim=32*2)


    ## Output layer
    self.conv2d_out = Conv2d(
      32, 2, (1, 1), (1, 1), padding=(0, 0), bias=False, 
      scope='conv2d_out')


    ## Kernels subject to l2-regularization
    update_dict(self.weights, self.renc1.weights)
    update_dict(self.weights, self.renc2.weights)
    update_dict(self.weights, self.renc3.weights)
    update_dict(self.weights, self.renc4.weights)
    update_dict(self.weights, self.ienc1.weights)
    update_dict(self.weights, self.ienc2.weights)
    update_dict(self.weights, self.ienc3.weights)
    update_dict(self.weights, self.ienc4.weights)
    update_dict(self.weights, self.rconv_mid.weights)
    update_dict(self.weights, self.iconv_mid.weights)
    update_dict(self.weights, self.dec4.weights)
    update_dict(self.weights, self.dec3.weights)
    update_dict(self.weights, self.dec2.weights)
    update_dict(self.weights, self.dec1.weights)
    update_dict(self.weights, self.conv2d_out.weights)
    self.weights = {scope+'/'+k:v for k,v in self.weights.items()}
    # self.weights_list = list(self.weights.values())
    self.weights_name = list(self.weights.keys())
    self.weights_list = [self.weights[name] for name in self.weights_name]

    ## Input normalization method
    self.input_norm_method = input_norm
    if input_norm == 'batchnorm':
      self.input_norm_r = nn.BatchNorm1d(input_dim)
      self.input_norm_i = nn.BatchNorm1d(input_dim)
    elif input_norm =='globalmvn':
      if mean_in is None:
        mean_in = np.zeros((1, input_dim, 1, 2), dtype='float32')
      if std_in is None:
        std_in = np.ones((1, input_dim, 1, 2), dtype='float32')
      self.input_norm = GMVN(mean_in, std_in)

    ## Output normalization method
    if mean_out is None:
      mean_out = np.zeros((1, input_dim, 1, 2), dtype='float32')
    if std_out is None:
      std_out = np.ones((1, input_dim, 1, 2), dtype='float32')
    self.output_norm = GMVN(mean_out, std_out)

    ## Loss functions for training this model
    self.loss_mse = nn.MSELoss(reduction='mean')
    self.loss_mae = nn.L1Loss(reduction='mean')

  @staticmethod
  def parse_batch(data, target, device):
    data, target = to_gpu(data, device), to_gpu(target, device)
    return data, target

  def forward(self, x, drop=0.0):
    ## If the input is batched waveforms, compute STFT coefficients;
    ## otherwise, the input must be STFT coefficients.
    if x.dim() < 3:
      x = self.stft_helper.stft(x) # (B, F, T, 2)

    ## Input normalization
    # x = self.input_norm(x)
    if self.input_norm_method == 'globalmvn':
      x = self.input_norm(x)
    else:
      x = torch.stack(
        (self.input_norm_r(x[..., 0]), self.input_norm_i(x[..., 1])), 
        dim=-1)

    ## Transpose to pass to the Conv2D layers
    x = x.permute(0, 3, 1, 2) # (B, C=2, F, T)

    ## First encoder stream
    renc1_hd, renc1_h2 = self.renc1(x, drop=drop)
    renc2_hd, renc2_h2 = self.renc2(renc1_hd, drop=drop)
    renc3_hd, renc3_h1 = self.renc3(renc2_hd, drop=drop)
    renc4_hd, renc4_h1 = self.renc4(renc3_hd, drop=drop)
    ## Second encoder stream
    ienc1_hd, ienc1_h2 = self.ienc1(x, drop=drop)
    ienc2_hd, ienc2_h2 = self.ienc2(ienc1_hd, drop=drop)
    ienc3_hd, ienc3_h1 = self.ienc3(ienc2_hd, drop=drop)
    ienc4_hd, ienc4_h1 = self.ienc4(ienc3_hd, drop=drop)

    ## Bottleneck Conv2D layers
    rmid = self.rconv_mid(renc4_hd)
    imid = self.iconv_mid(ienc4_hd)

    ## Decoder stream
    enc4_hd_ishape = renc4_h1.size()[:1]+(self.du4,)+renc4_h1.size()[2:]
    # print('enc4_hd_ishape', enc4_hd_ishape)
    dec4_h1 = self.dec4(
      input=torch.cat((rmid, imid), dim=1), output_size=enc4_hd_ishape, 
      input_aux=torch.cat((renc4_h1, ienc4_h1), dim=1), drop=drop)

    enc3_hd_ishape = renc3_h1.size()[:1]+(self.du3,)+renc3_h1.size()[2:]
    # print('enc3_hd_ishape', enc3_hd_ishape)
    dec3_h1 = self.dec3(
      input=dec4_h1, output_size=enc3_hd_ishape, 
      input_aux=torch.cat((renc3_h1, ienc3_h1), dim=1), drop=drop)

    enc2_hd_ishape = renc2_h2.size()[:1]+(self.du2,)+renc2_h2.size()[2:]
    # print('enc2_hd_ishape', enc2_hd_ishape)
    dec2_h1 = self.dec2(
      input=dec3_h1, output_size=enc2_hd_ishape, 
      input_aux=torch.cat((renc2_h2, ienc2_h2), dim=1), drop=drop)

    enc1_hd_ishape = renc1_h2.size()[:1]+(self.du1,)+renc1_h2.size()[2:]
    # print('enc1_hd_ishape', enc1_hd_ishape)
    dec1_h1 = self.dec1(
      input=dec2_h1, output_size=enc1_hd_ishape, 
      input_aux=torch.cat((renc1_h2, ienc1_h2), dim=1), drop=drop)

    ## Output 1x1 Conv2D layer
    v = self.conv2d_out(dec1_h1) # (B, 2, F, T)
    v = v.permute(0, 2, 3, 1)    # (B, F, T, 2)

    ## Fake scaling
    v = v / self.fake_scale

    ## Output normalization
    v = self.output_norm.denormalize(v)
    return v

  def get_loss(self, sig_x, sig_y, alpha, beta, gamma, drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_y" are batched time-domain waveforms """
    stft_v = self.forward(sig_x, drop=drop) # (B, F, T, 2)
    stft_y = self.stft_helper.stft(sig_y)  # (B, F, T, 2)
    lms_v, lms_y = spo.stft2lms(stft_v), spo.stft2lms(stft_y) # (B, F, T)
    sig_v = self.stft_helper.istft(stft_v, length=sig_x.size(1)) # (B, T)
    # sig_y = self.stft_helper.istft(stft_y, length=sig_x.size(1)) # (B, T)

    mse_stft_r_vy = self.loss_mse(stft_v[..., 0], stft_y[..., 0])
    mse_stft_i_vy = self.loss_mse(stft_v[..., 1], stft_y[..., 1])
    mse_lms_vy = self.loss_mse(lms_v, lms_y)
    mae_wav_vy = self.loss_mae(sig_v, sig_y)

    raw_loss = alpha*(mse_stft_r_vy+mse_stft_i_vy) \
      + beta*mse_lms_vy + gamma*mae_wav_vy
    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_vy":mse_stft_r_vy.item(), 
        "mse_stft_i_vy":mse_stft_i_vy.item(), 
        "mse_lms_vy":mse_lms_vy.item(), 
        "mae_wav_vy":mae_wav_vy.item(), 
      }
      return raw_loss, loss_dict, \
        (lms_y[-1], lms_v[-1], stft_y[-1], stft_v[-1], sig_y[-1], sig_v[-1])

  def get_loss_v2(self, sig_x, sig_y, alpha, beta, gamma, delta, drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_y" are batched time-domain waveforms """
    stft_v = self.forward(sig_x, drop=drop) # (B, F, T, 2)
    stft_y = self.stft_helper.stft(sig_y)  # (B, F, T, 2)
    mag_v, lms_v = spo.stft2maglms(stft_v) # (B, F, T)
    mag_y, lms_y = spo.stft2maglms(stft_y) # (B, F, T)
    sig_v = self.stft_helper.istft(stft_v, length=sig_x.size(1)) # (B, T)
    # sig_y = self.stft_helper.istft(stft_y, length=sig_x.size(1)) # (B, T)

    mse_stft_r_vy = self.loss_mse(stft_v[..., 0], stft_y[..., 0])
    mse_stft_i_vy = self.loss_mse(stft_v[..., 1], stft_y[..., 1])
    mse_lms_vy = self.loss_mse(lms_v, lms_y)
    mae_wav_vy = self.loss_mae(sig_v, sig_y)
    mae_mag_vy = self.loss_mae(mag_v, mag_y)

    raw_loss = alpha*(mse_stft_r_vy+mse_stft_i_vy) \
      + beta*mse_lms_vy + gamma*mae_wav_vy + delta*mae_mag_vy
    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_vy":mse_stft_r_vy.item(), 
        "mse_stft_i_vy":mse_stft_i_vy.item(), 
        "mse_lms_vy":mse_lms_vy.item(), 
        "mae_wav_vy":mae_wav_vy.item(), 
        "mae_mag_vy":mae_mag_vy.item(), 
      }
      return raw_loss, loss_dict, \
        (lms_y[-1], lms_v[-1], stft_y[-1], stft_v[-1], sig_y[-1], sig_v[-1])

