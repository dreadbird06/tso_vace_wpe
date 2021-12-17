"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_custom.torch_utils import shape, to_gpu, to_arr
from torch_custom.custom_layers import CustomModel, update_dict

from torch_custom.stft_helper import StftHelper
import torch_custom.spectral_ops as spo
from torch_custom.spectral_ops import MelFeatureExtractor

from torch_custom.wpe_th_utils import wpe_mb_torch_ri


# class VACEWPE(nn.Module):
class VACEWPE(CustomModel):
  def __init__(self, stft_opts, lpsnet=None, vacenet=None, mfcc_opts={}):
    super(VACEWPE, self).__init__()
    assert lpsnet is not None and isinstance(lpsnet, nn.Module)
    assert vacenet is not None and isinstance(vacenet, nn.Module)

    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)

    self.lpsnet = lpsnet
    self.vacenet = vacenet

    self.weights = vacenet.weights
    self.weights_list = vacenet.weights_list
    self.weights_name = vacenet.weights_name

    ## Loss functions for training this model
    self.loss_mse = nn.MSELoss(reduction='mean')
    self.loss_mae = nn.L1Loss(reduction='mean')

    ## MFCC loss
    if len(mfcc_opts):
      self.melfeatext = MelFeatureExtractor(**mfcc_opts)

  def train(self):
    self.vacenet.train()

  def eval(self):
    self.vacenet.eval()
    self.lpsnet.eval()

  # def to(self, device):
  #   self.vacenet.to(device)
  #   self.lpsnet.to(device)

  @staticmethod
  def parse_batch(data, target, device):
    data, target = to_gpu(data, device), to_gpu(target, device)
    return data, target

  def forward(self, sig_x, delay=3, taps=10, drop=0.0):
    """ sig_x is batched single-channel time-domain waveforms 
        shape: (B, T) == (batch, time)
    """
    ## Convert the time-domain signal to the STFT coefficients
    nb, nt = sig_x.size() # (B,t)
    stft_x = self.stft_helper.stft(sig_x) # (B,F,T,2)

    ## Compute early PSD using the LPSNet
    lps_x = spo.stft2lps(stft_x) # (B,F,T)
    psd_x = self.lpsnet(lps_x, drop=drop).exp() # (B,F,T)

    ## Compute virtual signal using the VACENet
    stft_v = self.vacenet(stft_x, drop=drop) # (B,F,T,2)

    ## Stack the pair of actual and virtual signals
    stft_xv = torch.stack((stft_x, stft_v), dim=1) # (B,C=2,F,T,2)

    ## Batch-mode WPE
    ## >> STFT and PSD must be in shape (B,C,F,T,2) and (B,F,T), respectively.
    nfreq, nfrm = psd_x.size(1), psd_x.size(2)
    stft_wpe = wpe_mb_torch_ri(
      stft_xv, psd_x, taps=taps, delay=delay) # (B,C=2,F,T,2)

    ## Inverse STFT
    stft_wpe_x, stft_wpe_v = stft_wpe[:,0], stft_wpe[:,1] # (B,F,T,2)
    sig_wpe_x = self.stft_helper.istft(stft_wpe_x, length=nt) # (B,t)
    return sig_wpe_x, stft_wpe_x, lps_x, stft_v

  def dereverb(self, sig_x, delay=3, taps=10):
    sig_wpe_x = self.forward(sig_x, delay, taps)[0]
    return to_arr(sig_wpe_x).squeeze()
    return self.forward(sig_x, delay, taps)[0]

  def get_loss(self, sig_x, sig_early, delay, taps, 
               alpha, beta, gamma, drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_early" are batched time-domain waveforms """
    sig_wpe_x, stft_wpe_x, lps_x, stft_v = \
      self.forward(sig_x, delay, taps, drop=drop) # (B,t)
    # stft_wpe_x = self.stft_helper.stft(sig_wpe_x) # (B,F,T,2)
    stft_early = self.stft_helper.stft(sig_early) # (B,F,T,2)
    lms_wpe_x = spo.stft2lms(stft_wpe_x) # (B,F,T)
    lms_early = spo.stft2lms(stft_early) # (B,F,T)

    mse_stft_r_wpe = self.loss_mse(stft_wpe_x[..., 0], stft_early[..., 0])
    mse_stft_i_wpe = self.loss_mse(stft_wpe_x[..., 1], stft_early[..., 1])
    mse_lms_wpe = self.loss_mse(lms_wpe_x, lms_early)
    mae_wav_wpe = self.loss_mae(sig_wpe_x, sig_early)

    raw_loss = alpha*(mse_stft_r_wpe+mse_stft_i_wpe) \
             + beta*mse_lms_wpe + gamma*mae_wav_wpe

    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_wpe":mse_stft_r_wpe.item(), 
        "mse_stft_i_wpe":mse_stft_i_wpe.item(), 
        "mse_lms_wpe":mse_lms_wpe.item(), 
        "mae_wav_wpe":mae_wav_wpe.item(), 
      }
      return raw_loss, loss_dict, (
        0.5*lps_x[-1],                                            # lms_x
        spo.stft2lms(stft_v[-1]),                                 # lms_v
        lms_wpe_x[-1],                                            # lms_wpe_x
        sig_x[-1],                                                # sig_x
        self.stft_helper.istft(stft_v[-1], length=sig_x.size(1)), # sig_v
        sig_wpe_x[-1])                                            # sig_wpe_x

  def get_loss_mfcc(self, sig_x, sig_early, delay, taps, 
                    alpha, beta, gamma, delta=0.0, power_scale=True, 
                    drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_early" are batched time-domain waveforms """
    sig_wpe_x, stft_wpe_x, lps_x, stft_v = \
      self.forward(sig_x, delay, taps, drop=drop) # (B,t)
    # stft_wpe_x = self.stft_helper.stft(sig_wpe_x) # (B,F,T,2)
    stft_early = self.stft_helper.stft(sig_early) # (B,F,T,2)
    lms_wpe_x = spo.stft2lms(stft_wpe_x) # (B,F,T)
    lms_early = spo.stft2lms(stft_early) # (B,F,T)
    mfcc_wpe_x = self.melfeatext.mfcc(stft_wpe_x, power_scale=power_scale)
    mfcc_early = self.melfeatext.mfcc(stft_early, power_scale=power_scale)

    mse_stft_r_wpe = self.loss_mse(stft_wpe_x[..., 0], stft_early[..., 0])
    mse_stft_i_wpe = self.loss_mse(stft_wpe_x[..., 1], stft_early[..., 1])
    mse_lms_wpe = self.loss_mse(lms_wpe_x, lms_early)
    mae_wav_wpe = self.loss_mae(sig_wpe_x, sig_early)
    mae_mfcc_wpe = self.loss_mae(mfcc_wpe_x, mfcc_early)

    raw_loss = alpha*(mse_stft_r_wpe+mse_stft_i_wpe) \
             + beta*mse_lms_wpe + gamma*mae_wav_wpe + delta*mae_mfcc_wpe

    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_wpe":mse_stft_r_wpe.item(), 
        "mse_stft_i_wpe":mse_stft_i_wpe.item(), 
        "mse_lms_wpe":mse_lms_wpe.item(), 
        "mae_wav_wpe":mae_wav_wpe.item(), 
        "mae_mfcc_wpe":mae_mfcc_wpe.item(), 
      }
      return raw_loss, loss_dict, (
        0.5*lps_x[-1],                                            # lms_x
        spo.stft2lms(stft_v[-1]),                                 # lms_v
        lms_wpe_x[-1],                                            # lms_wpe_x
        sig_x[-1],                                                # sig_x
        self.stft_helper.istft(stft_v[-1], length=sig_x.size(1)), # sig_v
        sig_wpe_x[-1])                                            # sig_wpe_x



if __name__=="__main__":
  import os
  # os.environ["CUDA_VISIBLE_DEVICES"]="0"

  from gcunet4c_4M4390 import VACENet
  from bldnn_4M62 import LstmDnnNet as LPSEstimator

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print('device = {}'.format(device))
  # device = "cpu"


  ## STFT options
  stft_opts_torch = dict(
    n_fft=1024, hop_length=256, win_length=1024, win_type='hanning', 
    symmetric=True)
  fft_bins = stft_opts_torch['n_fft']//2 + 1

  mfcc_opts_torch = dict(
    fs=16000, nfft=1024, lowfreq=20., maxfreq=7600., 
    nlinfilt=0, nlogfilt=40, nceps=40, 
    lifter_type='sinusoidal', lift=-22.0) # only useful during fine-tuning

  ## Input tensors
  sig_x = torch.randn(6, 44800).to(device) # assume a mini-batch of waveforms
  sig_early = torch.randn(6, 44800).to(device)


  ## VACEMet
  vacenet = VACENet(
    input_dim=fft_bins, stft_opts=stft_opts_torch, scope='vace_unet')
  vacenet.to(device)
  print('VACENet size', vacenet.size)
  print(len(tuple(vacenet.trainable_parameters())))
  print(len(tuple(vacenet.parameters())))

  ## LPSNet
  fft_bins = stft_opts_torch['n_fft']//2 + 1
  lpsnet = LPSEstimator(
    input_dim=fft_bins, stft_opts=stft_opts_torch, scope='ldnn_lpseir_ns')
  lpsnet.to(device)
  print('LPSNet size', lpsnet.size)
  print(len(tuple(lpsnet.trainable_parameters())))
  print(len(tuple(lpsnet.parameters()))); #exit()

  ## Freeze the LPSNet
  # lpsnet.check_trainable_parameters()
  lpsnet.freeze()
  # lpsnet.check_trainable_parameters()
  # lpsnet.unfreeze()
  # lpsnet.check_trainable_parameters()
  lpsnet.eval()


  ## VACE-WPE
  vace_wpe = VACEWPE(
    stft_opts=stft_opts_torch, 
    lpsnet=lpsnet, vacenet=vacenet, 
    mfcc_opts=mfcc_opts_torch)
  vace_wpe.to(device)
  print('VACE-WPE size', vace_wpe.size)
  vace_wpe.check_trainable_parameters()
  print(len(tuple(vace_wpe.trainable_parameters())))
  print(len(tuple(vace_wpe.parameters())))
  for name in vace_wpe.weights_name: print(name)
  kernel_l2_norm, kernel_l2_norm_dict = vace_wpe.weights_l2_norm()
  print('weight_l2_norm = %.5f' % kernel_l2_norm.item())


  ## Compute loss
  loss_scales = dict(alpha=1.0, beta=0.1, gamma=5.0, delta=0.2, power_scale=True)
  loss = vace_wpe.get_loss_mfcc(
    sig_x, sig_early, delay=3, taps=5, 
    **loss_scales ,drop=0.0, summarize=False)
  ## L2-regularization
  total_loss = loss + (1e-5)*kernel_l2_norm
  ## Compute gradients
  total_loss.backward()
  print('Succeeded!!')

