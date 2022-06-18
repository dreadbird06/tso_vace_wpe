"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch
import torch.nn as nn

from torch_custom.stft_helper import StftHelper
import torch_custom.spectral_ops as spo
# from torch_custom.custom_layers import CustomModel

from torch_custom.wpe_th_utils import wpe_mb_torch


# class IterativeWPE(CustomModel):
class IterativeWPE(nn.Module):
  def __init__(self, stft_opts):
    super(IterativeWPE, self).__init__()
    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)

  def forward(self, sig_x, delay=3, taps=10, n_iter=1, dtype=torch.float32):
    """ sig_x is batched multi-channel time-domain waveforms 
        shape: (B, C, T) == (batch, channels, time)
    """
    ## Convert the time-domain signal to the STFT coefficients
    nb, nc, nt = sig_x.size() # (B,C,t)
    sig_x = sig_x.view(nb*nc, nt) # (BC,t)
    stft_x = self.stft_helper.stft(sig_x) # (BC,F,T(,2))
    nfreq, nfrm = stft_x.size(1), stft_x.size(2)
    if torch.is_complex(stft_x):
      dtype = torch.complex64
    stft_x = stft_x.view(nb, nc, *stft_x.size()[1:]).contiguous().to(dtype) # (B,C,F,T,2)

    ## Start iterations
    stft_z = stft_x.clone()
    for it in range(n_iter):
      ## Compute PSD
      psd_z = spo.stft2ps(stft_z) # (B,C,F,T)
      psd_z_mean = psd_z.mean(dim=1) # (B,C,F,T) >> (B,F,T)

      ## Batch-mode WPE
      ## >> STFT and PSD must be in shape (B,C,F,T(,2)) and (B,F,T), respectively.
      stft_z = wpe_mb_torch(
        stft_x, psd_z_mean, taps=taps, delay=delay) # (B,C,F,T(,2))
    if not torch.is_complex(stft_z):
      stft_z = stft_z.float()

    ## Inverse STFT
    stft_z = stft_z.view(nb*nc, *stft_z.size()[2:]) # (BC,F,T(,2))
    sig_z = self.stft_helper.istft(stft_z, length=nt) # (BC,t)
    sig_z = sig_z.view(nb, nc, nt) # (B,C,t)
    return sig_z
