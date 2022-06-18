"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import numpy as np

import torch
import torch.nn as nn

from torch_custom.stft_helper import StftHelper


def hz2mel(f, htk=True):
  """ https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py """
  if htk:
    return 1127. * np.log(1. + f / 700.)
    return 2595 * np.log10(1 + f / 700.)
  else:
    f = np.array(f)
    ## Mel fn to match Slaney's Auditory Toolbox mfcc.m
    f_0 = 0.
    f_sp = 200. / 3.
    brkfrq = 1000.
    brkpt  = (brkfrq - f_0) / f_sp
    logstep = np.exp(np.log(6.4) / 27)

    linpts = f < brkfrq

    z = np.zeros_like(f)
    ## fill in parts separately
    z[linpts] = (f[linpts] - f_0) / f_sp
    z[~linpts] = brkpt + (np.log(f[~linpts] / brkfrq)) / np.log(logstep)

    if z.shape == (1,):
      return z[0]
    else:
      return z

def mel2hz(z, htk=True):
  """ https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py """
  if htk:
    return 700. * (np.exp(z / 1127.) - 1.)
    return 700. * (10**(z / 2595.) - 1)
  else:
    z = np.array(z, dtype=float)
    f_0 = 0
    f_sp = 200. / 3.
    brkfrq = 1000.
    brkpt  = (brkfrq - f_0) / f_sp
    logstep = np.exp(np.log(6.4) / 27)

    linpts = (z < brkpt)

    f = np.zeros_like(z)
    # fill in parts separately
    f[linpts] = f_0 + f_sp * z[linpts]
    f[~linpts] = brkfrq * np.exp(np.log(logstep) * (z[~linpts] - brkpt))

    if f.shape == (1,):
      return f[0]
    else:
      return f

def trfbank_np(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
  """ https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py 
      (slightly modified)
  """
  maxfreq = min(maxfreq, fs/2)
  ## Total number of filters
  nfilt = nlinfilt + nlogfilt

  ## 1. Compute start/middle/end points of the triangular filters in spectral domain
  ## Use only "linearly-scaled" filter banks
  if nlogfilt == 0:
    freqs = lowfreq + np.arange(nlinfilt+2) * (maxfreq-lowfreq) / (nlinfilt+1)

  ## Use only "log-linearly-scaled" filter banks (*linear in log-domain)
  elif nlinfilt == 0:
    low_mel = hz2mel(lowfreq, htk=True)
    max_mel = hz2mel(maxfreq, htk=True)
    mels = low_mel + np.arange(nlogfilt+2) * (max_mel-low_mel) / (nlogfilt+1)
    freqs = mel2hz(mels)  # non-linear in lin-scale

  ## Use linear-scale below midfreq, log-linear-scale above midfreq
  else:
    ## Compute linear filters on [0;1000Hz]
    freqs = np.zeros(nfilt + 2, dtype='float32')
    linscale = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linscale
    ## Compute log-linear filters on [1000;maxfreq]
    low_mel = hz2mel(min([1000, maxfreq]), htk=True)
    max_mel = hz2mel(maxfreq, htk=True)
    mels = np.zeros(nlogfilt + 2, dtype='float32')
    melscale = (max_mel - low_mel) / (nlogfilt + 1)

    ## Verify that mel2hz(melscale)>linscale
    while mel2hz(melscale) < linscale:
      # in this case, we add a linear filter
      nlinfilt += 1
      nlogfilt -= 1
      freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linscale
      low_mel = hz2mel(freqs[nlinfilt - 1] + 2 * linscale, htk=True)
      max_mel = hz2mel(maxfreq, htk=True)
      # mels = np.zeros(nlogfilt + 2, dtype='float32')
      melscale = (max_mel - low_mel) / (nlogfilt + 1)

    mels[:nlogfilt + 2] = low_mel + np.arange(nlogfilt + 2) * melscale  #@@@ linear in mel-scale
    freqs[nlinfilt:] = mel2hz(mels)

  ## 2. Compute filterbank coeff (in fft domain, in bins)
  fbank = np.zeros((nfilt, int(np.floor(nfft / 2)) + 1), dtype='float32')
  fbin_width = (fs / float(nfft)) * np.arange(nfft)  # equispaced frequency bins (in Hz)
  mbin_width = hz2mel(fbin_width, htk=True)
  for i in range(nfilt):
    low, cen, hi = freqs[i:i+2+1]
    lid = np.arange(np.floor(low*nfft/fs)+1, np.floor(cen*nfft/fs)+1, dtype='int32')
    rid = np.arange(np.floor(cen*nfft/fs)+1, min(np.floor(hi*nfft/fs)+1, nfft), dtype='int32')
    aid = np.append(lid, rid)

    low, cen, hi = mels[i:i+2+1]
    left_slope = (mbin_width[lid] - low) / (cen - low)
    right_slope = (hi - mbin_width[rid]) / (hi - cen)

    trifilt = np.append(left_slope, right_slope)
    fbank[i,aid] = trifilt
  return fbank, freqs


def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, dtype='float32'):
  """ triangular filterbank """
  fbank, _ = trfbank_np(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt)
  return torch.from_numpy(fbank.astype(dtype))

def cepstral_lifter(nceps, type='sinusoidal', lift=-22.0, inverse=False):
  """ https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py 
      (slightly modified)
  """
  if lift == 0:
    # print('Do not apply cepstral lifting (multiply an identity matrix)...')
    return torch.eye(nceps).float()
  else:
    if type == 'linear':
      # print('Use a linear lifter...')
      liftwts = np.arange(nceps)

    elif type == 'exponential':
      # print('Use an exponential lifter...')
      s, tau = 1.5, 5
      qfbins = np.arange(nceps)
      liftwts = qfbins**s * np.exp(-qfbins**2/2/tau**2)

    elif type == 'sinusoidal':
      # print('Use a sinusoidal lifter...')

      if lift > 0:
        if lift > 10:
          print('Unlikely lift exponent of {} did you mean -ve?'.format(lift))
        liftwts = np.hstack((1, np.arange(1, nceps)**lift))

      elif lift < 0:  # Hack to support HTK liftering
        L = float(-lift)
        if (L != np.round(L)):
          print('HTK liftering value {} must be integer'.format(L))
        liftwts = np.hstack((1, 1 + L/2*np.sin(np.arange(1, nceps) * np.pi / L)))

      if inverse:
        liftwts = 1 / liftwts

    liftwts_diag = np.diag(liftwts).astype('float32')
    return torch.from_numpy(liftwts_diag)


""" https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py """
def dct_type2(x, norm=None):
  """
  Discrete Cosine Transform, Type II (a.k.a. the DCT)
  For the meaning of the parameter `norm`, see:
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
  :param x: the input signal
  :param norm: the normalization, None or 'ortho'
  :return: the DCT-II of the signal over the last dimension
  """
  x_shape = x.size()
  N = x_shape[-1]
  x = x.contiguous().view(-1, N)

  v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
  if torch.__version__ < '1.8':
    Vc = torch.rfft(v, 1, onesided=False)
    Vc_r, Vc_i = Vc[..., 0], Vc[..., 1]
  else:
    # Vc = torch.fft.fft(v, n=v.size(-1), dim=-1)
    Vc = torch.fft.fft(v, dim=-1)
    Vc_r, Vc_i = Vc.real, Vc.imag

  k = -torch.arange(N, dtype=x.dtype, device=x.device)[None] * np.pi / (2*N)
  W_r, W_i = torch.cos(k), torch.sin(k)

  V = Vc_r*W_r - Vc_i*W_i
  if norm == 'ortho':
    V[:, 0] /= np.sqrt(N) * 2
    V[:, 1:] /= np.sqrt(N/2) * 2
  return 2 * V.view(*x_shape)


class MelFeatureExtractor(nn.Module):
  def __init__(self, fs=16000, nfft=1024, lowfreq=20., maxfreq=7600., 
               nlinfilt=0, nlogfilt=40, nceps=40, 
               lifter_type='sinusoidal', lift=-22.0):
    super(MelFeatureExtractor, self).__init__()

    melfb = trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt) # (M,F)
    lifter = cepstral_lifter(nceps, type=lifter_type, lift=lift, inverse=False)
    self.register_buffer('melfb', melfb)
    self.register_buffer('lifter', lifter)
    self.nceps = nceps

  def mfbe(self, coefs, power_scale=True, apply_log=True, eps=1.19209e-07):
    """ coef in shape (B,F,T,2) """
    ## Convert STFT to magnitude spectra
    magspec = stft2mag(coefs) # (B,F,T)
    ## Power spectra
    if power_scale:
      magspec = torch.pow(magspec, 2) # (B,F,T)
    ## Mel-spectra
    melspec = torch.einsum('bft,mf->bmt', 
      magspec, self.melfb.to(coefs.device)).clamp_(min=eps)
    ## Log-scale
    if apply_log:
      return torch.log(melspec.float())
    ## Linear-scale
    return melspec.float()

  def liftering(self, cepstra):
    return torch.einsum('bft,fg->bgt', cepstra, self.lifter)

  def mfcc(self, coefs, power_scale=True, eps=1.19209e-07):
    """ mel-frequency cepstral coefficients """
    melspec = self.mfbe(coefs, power_scale, apply_log=True, eps=eps)
    melceps = dct_type2(melspec.transpose(-2, -1), norm='ortho').transpose(-2, -1)
    return self.liftering(melceps[:,:self.nceps,:])


class MelFeatureExtractorV2(MelFeatureExtractor):
  def __init__(self, fs=16000, nfft=1024, lowfreq=20., maxfreq=7600., 
               nlinfilt=0, nlogfilt=40, nceps=40, lift=-22.0):
    super(MelFeatureExtractorV2, self).__init__(
      fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, nceps, lift)

    import torchaudio as ta
    dctmat = ta.functional.create_dct(
      n_mfcc=nceps, n_mels=nlogfilt-nlinfilt, norm='ortho')
    self.register_buffer('dctmat', dctmat)
    # print('DCT-TypeII matrix:', list(dctmat.size()))

  def mfcc(self, coefs, power_scale=True, eps=1.19209e-07):
    """ mel-frequency cepstral coefficients """
    melspec = self.mfbe(coefs, power_scale, apply_log=True, eps=eps)
    melceps = torch.einsum('bit,ij->bjt', melspec, self.dctmat)
    return self.liftering(melceps[:,:self.nceps,:])


class SpectralFeatureExtractor(nn.Module):
  def __init__(self, stft_opts={}, mel_opts={}, nceps=-1, lift=-22.0):
    super(SpectralFeatureExtractor, self).__init__()

    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)

    nfft = stft_opts['n_fft']
    if mel_opts is not None and len(mel_opts):
      melfb = trfbank(**mel_opts)#fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt) # (M,F)
      self.mel_dim = melfb.size(0)
      self.register_buffer('melfb', melfb)
    if nceps:
      lifter = cepstral_lifter(nceps, lift=lift, inverse=False)
      self.register_buffer('lifter', lifter)
      self.nceps = nceps

  def stft(self, wave, center=True):
    """ wave in shape (B,t) """
    return self.stft_helper.stft(wave, center=center) # (B,F,T,2)

  def istft(self, coefs, length=None):
    return self.stft_helper.istft(coefs, length=length)

  def mfbe(self, wave, power_scale=True, apply_log=True, eps=1.19209e-07, center=True):
    """ wave in shape (B,t) """
    ## Compute STFT coefficients
    if wave.dim() < 3:
      coefs = self.stft_helper.stft(wave, center=center) # (B, F, T, 2)
    else:
      coefs = wave
    ## Convert STFT to magnitude spectra
    magspec = stft2mag(coefs) # (B,F,T)
    ## Power spectra
    if power_scale:
      magspec = torch.pow(magspec, 2) # (B,F,T)
    ## Mel-spectra
    melspec = torch.einsum('bft,mf->bmt', 
      magspec, self.melfb.to(coefs.device)).clamp_(min=eps)
    ## Log-scale
    if apply_log:
      return torch.log(melspec.float())
    ## Linear-scale
    return melspec.float()

  def liftering(self, cepstra):
    return torch.einsum('bft,fg->bgt', cepstra, self.lifter)

  def mfcc(self, wave, power_scale=True, eps=1.19209e-07, center=True):
    """ mel-frequency cepstral coefficients """
    melspec = self.mfbe(wave, power_scale, apply_log=True, eps=eps, center=center)
    melceps = dct_type2(melspec.transpose(-2, -1), norm='ortho').transpose(-2, -1)
    return self.liftering(melceps[:,:self.nceps,:])

  def mfbe2mfcc(self, melspec):
    melceps = dct_type2(melspec.transpose(-2, -1), norm='ortho').transpose(-2, -1)
    return self.liftering(melceps[:,:self.nceps,:])



def polar2rect(mag, pha, return_complex=False):
  coefs_r, coefs_i = mag*pha.cos(), mag*pha.sin()
  if return_complex:
    return torch.complex(coefs_r, coefs_i)
  else:
    return coefs_r, coefs_i

def polar2stft(mag, pha, return_complex=False):
  coefs_r, coefs_i = polar2rect(mag, pha)
  if return_complex:
    return torch.complex(coefs_r, coefs_i)
  else:
    return torch.stack((coefs_r, coefs_i), dim=-1)

def stft2polar(coefs):
  if torch.is_complex(coefs):
    coefs = torch.view_as_real(coefs)
  coefs_r, coefs_i = coefs[..., 0], coefs[..., 1]
  mag = torch.sqrt(coefs_r**2 + coefs_i**2)
  pha = torch.atan2(coefs_i, coefs_r)
  return mag, pha

def stft2rect(coefs):
  if torch.is_complex(coefs):
    coefs = torch.view_as_real(coefs)
  coefs_r, coefs_i = coefs[..., 0], coefs[..., 1]
  return coefs_r, coefs_i

def stft2mag(coefs):
  if torch.is_complex(coefs):
    return coefs.abs()
  else:
    return torch.sqrt(coefs[..., 0]**2 + coefs[..., 1]**2)

def stft2pha(coefs, eps=1e-7):
  if torch.is_complex(coefs):
    return torch.atan2(coefs.imag, coefs.real)
  else:
    return torch.atan2(coefs[..., 1], coefs[..., 0])

def stft2lms(coefs, eps=1.19209e-07):
  if torch.is_complex(coefs):
    mag = coefs.abs()
  else:
    mag = torch.sqrt(coefs[..., 0]**2 + coefs[..., 1]**2)
  return torch.log(mag + eps)

def stft2maglms(coefs, eps=1.19209e-07):
  mag = stft2mag(coefs)
  return mag, torch.log(mag + eps)

def stft2lps(coefs, eps=1.19209e-07):
  if torch.is_complex(coefs):
    pspec = coefs.abs().square()
  else:
    pspec = coefs[..., 0]**2 + coefs[..., 1]**2
  return torch.log(pspec + eps)

def stft2ps(coefs):
  if torch.is_complex(coefs):
    return coefs.real.square() + coefs.imag.square()
  else:
    return coefs[..., 0]**2 + coefs[..., 1]**2

def stft2cms(coefs, alpha=0.3):
  """ Power-law compressed magnitude spectrum """
  return stft2mag(coefs).pow_(alpha)

def cms2mag(cms, alpha=0.3):
  return cms.pow_(1/alpha)

