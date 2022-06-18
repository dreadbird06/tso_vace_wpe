"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torch_custom.torch_utils import to_arr


# def signal_framing(signal, frame_length, frame_step, dim=-1):
#   """ Framing with stride tricks
#       (https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py)
#   """
#   shape = list(signal.size())
#   nframes = (shape[-1] - frame_length + frame_step) // frame_step
#   if nframes > 0:
#     shape = shape[:-1] + [nframes, frame_length]
#     strides = list(signal.stride())
#     strides.insert(-1, frame_step*strides[-1])
#     return torch.as_strided(signal, size=shape, stride=strides)
#   else:
#     return signal.unsqueeze(dim=-2)

def signal_framing(signal, frame_length, frame_step, dim=-1):
  """ Framing with stride tricks
      (https://git-lium.univ-lemans.fr/Larcher/sidekit/-/blob/master/frontend/features.py)
  """
  dim = dim % signal.dim()
  shape = list(signal.size())
  nframes = (shape[dim] - frame_length + frame_step) // frame_step
  if nframes > 0:
    shape = shape[:dim] + [nframes, frame_length] + shape[dim+1:]
    strides = list(signal.stride())
    strides.insert(dim, frame_step*strides[dim])
    return torch.as_strided(signal, size=shape, stride=strides)
  else:
    return signal.unsqueeze(dim=dim)


def pad_for_stft(sig, winLen=400, center=True, pad_mode='constant'):
  """ https://pytorch.org/docs/stable/_modules/torch/functional.html#stft """
  if center:
    ndim = sig.dim()
    extended_shape = [1]*(3-ndim) + list(sig.size())
    pad = int(winLen//2)
    sig = F.pad(sig.view(extended_shape), pad=[pad, pad], mode=pad_mode)
    # sig = F.pad(sig.view(extended_shape), [pad, pad], 'constant', value=0)
    return sig.view(sig.shape[-ndim:])
  else:
    return sig


def torch_vad(sig, winLen=400, winSht=160, ener_thres=5.7, 
              mean_scale=0.5, prop_thres=0.12, frm_context=2, 
              twice_log_max_signed_int16=20.794354380710768, 
              eps=1.19209e-07, center=False):
  """ Equivalent to KALDI's energy-based VAD implementation 
      (https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/voice-activity-detection.cc)
  """
  ## Framing
  if center:
    sig = F.pad(sig, (winLen-winSht, winLen-winSht))
  framed = signal_framing(
    sig, frame_length=winLen, frame_step=winSht, dim=1)
  ## DC removal
  framed = framed.sub(framed.mean(dim=-1, keepdim=True))

  ## Compute signal power (averaged over channels)
  # framed = framed.mean(axis=0)  # channel average
  log_power = framed.square().sum(dim=-1).clamp_(eps).log_()
  # log_power = framed.square().sum(dim=-1).add_(eps).log_()
  if sig.max() < 1.0:
    log_power.add_(twice_log_max_signed_int16)

  ## Set VAD threshold
  if mean_scale:
    thres_offset = mean_scale*log_power.sum(dim=-1, keepdim=True) / log_power.size(-1)
    ener_thres = ener_thres + thres_offset

  ## Set context length & context window
  len_context = 2*frm_context + 1

  ## Compute VAD by counting over-thresh frames followed by thresholding those counts
  vad_s = log_power.gt_(ener_thres)
  vad_s = signal_framing(
    F.pad(vad_s, [frm_context, frm_context], mode='constant', value=0), 
    frame_length=len_context, frame_step=1, dim=-1).sum(dim=-1)
  vad_c = vad_s.ge(prop_thres * len_context)
  return vad_c

def plot_vad(sig, vad, winSht, uttid='vad'):
  if isinstance(sig, torch.Tensor):
    sig = to_arr(sig).squeeze()
  if isinstance(vad, torch.Tensor):
    vad = to_arr(vad).squeeze()

  plt.plot(np.arange(sig.shape[-1])/winSht, sig/np.abs(sig).max())
  plt.plot(np.arange(vad.shape[-1]), 0.1+0.5*vad)

def plot_vad_v2(sig, winSht, vad1, vad2=None, uttid='vad'):
  if isinstance(sig, torch.Tensor):
    sig = to_arr(sig).squeeze()
  if isinstance(vad1, torch.Tensor):
    vad1 = to_arr(vad1).squeeze()

  plt.plot(np.arange(sig.shape[-1])/winSht, sig/np.abs(sig).max(), 'b')
  plt.plot(np.arange(vad1.shape[-1]), vad1, 'r')
  if vad2 is not None:
    if isinstance(vad2, torch.Tensor):
      vad2 = to_arr(vad2).squeeze()
    plt.plot(np.arange(vad2.shape[-1]), vad2, 'g')
