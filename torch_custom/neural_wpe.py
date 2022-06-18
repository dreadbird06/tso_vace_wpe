"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch
import torch.nn as nn

from torch_custom.stft_helper import StftHelper
import torch_custom.spectral_ops as spo
from torch_custom.custom_layers import CustomModel

from torch_custom.wpe_th_utils import wpe_mb_torch


# class NeuralWPE(nn.Module):
class NeuralWPE(CustomModel):
  def __init__(self, stft_opts, lpsnet=None):
    super(NeuralWPE, self).__init__()
    assert lpsnet is not None and isinstance(lpsnet, nn.Module)
    if stft_opts is None:
      self.stft_helper = lpsnet.stft_helper
    else:
      assert len(stft_opts) >= 5
      self.stft_helper = StftHelper(**stft_opts)
    self.lpsnet = lpsnet
    self.weights = lpsnet.weights
    self.weights_list = lpsnet.weights_list
    self.weights_name = lpsnet.weights_name

  def train(self):
    self.lpsnet.train()

  def eval(self):
    self.lpsnet.eval()

  def forward(self, sig_x, delay=3, taps=10, drop=0.0, dtype=torch.float32, ref_ch=None):
    """ sig_x is batched multi-channel time-domain waveforms 
        shape: (B, C, T) == (batch, channels, time)
    """
    ## Convert the time-domain signal to the STFT coefficients
    nb, nc, nt = sig_x.size() # (B,C,t)
    sig_x = sig_x.view(nb*nc, nt) # (BC,t)
    stft_x = self.stft_helper.stft(sig_x) # (BC,F,T,2)

    ## Compute the PSD using a pre-trained neural network
    lps_x = spo.stft2lps(stft_x) # (BC,F,T)
    psd_x = self.lpsnet(lps_x, drop=drop).exp() # (BC,F,T)

    ## Batch-mode WPE
    ## >> STFT and PSD must be in shape (B,C,F,T,2) and (B,F,T), respectively.
    nfreq, nfrm = psd_x.size(1), psd_x.size(2)
    if torch.is_complex(stft_x):
      stft_x = stft_x.view(nb, nc, nfreq, nfrm).contiguous() # (B,C,F,T,2)
    else:
      stft_x = stft_x.view(nb, nc, nfreq, nfrm, 2).contiguous().type(dtype) # (B,C,F,T,2)
    psd_x_mean = psd_x.view(nb, nc, nfreq, nfrm).mean(dim=1) # (B,C,F,T) >> (B,F,T)
    stft_wpe = wpe_mb_torch(
      stft_x, psd_x_mean, taps=taps, delay=delay, ref_ch=ref_ch) # (B,C,F,T,2)
    if ref_ch is not None: nc = 1
    if torch.is_complex(stft_x):
      stft_wpe = stft_wpe.view(nb*nc, nfreq, nfrm) # (BC,F,T,2)
    else:
      stft_wpe = stft_wpe.float().view(nb*nc, nfreq, nfrm, 2) # (BC,F,T,2)

    ## Inverse STFT
    sig_wpe = self.stft_helper.istft(stft_wpe, length=nt) # (BC,t)
    sig_wpe = sig_wpe.view(nb, nc, nt) # (B,C,t)
    return sig_wpe
