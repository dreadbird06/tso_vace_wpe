"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch
from torch import nn

if torch.__version__ < '1.7':
  support_complex_tensor = False
else:
  support_complex_tensor = True

window_dict = {
  'hamming':torch.hamming_window, 'hanning':torch.hann_window, 
  'blackman':torch.blackman_window, 'bartlett':torch.bartlett_window, 
}


class StftHelper(nn.Module):
  def __init__(self, n_fft, hop_length, win_length, win_type, symmetric=False):
    super(StftHelper, self).__init__()

    ## Windowing options
    self.winLen = win_length
    self.winSht = hop_length
    # self.overlap = win_length - hop_length
    self.nfft = n_fft

    ## Analysis window
    win_type = win_type.lower()
    assert win_type in ('hamming', 'hanning', 'blackman')
    self.analy_window = window_dict[win_type](
      win_length, periodic=not symmetric)

  def stft(self, signal, time_axis=-1, return_complex=True, center=True):
    """ signal: (batch, samples) """
    ## Compute STFT coefficients
    if support_complex_tensor:
      coefs = torch.stft(signal, 
        n_fft=self.nfft, hop_length=self.winSht, win_length=self.winLen, 
        window=self.analy_window.to(signal.device), center=center, pad_mode='constant', 
        normalized=False, onesided=True, return_complex=return_complex)
    else:
      coefs = torch.stft(signal, 
        n_fft=self.nfft, hop_length=self.winSht, win_length=self.winLen, 
        window=self.analy_window.to(signal.device), center=center, pad_mode='constant', 
        normalized=False, onesided=True)
    return coefs # (batch, bins, frame(, 2))

  def istft(self, coefs, length=None, time_axis=-2):
    """ coefs: (batch, frames, bins(, 2)) """
    ## Compute inverse STFT
    if support_complex_tensor and not torch.is_complex(coefs):
        coefs = torch.complex(coefs[..., 0], coefs[..., 1])

    signal = torch.istft(coefs, 
      n_fft=self.nfft, hop_length=self.winSht, win_length=self.winLen, 
      window=self.analy_window.to(coefs.device), center=True, 
      normalized=False, onesided=True, 
      length=length)
    return signal
