"""
Example codes for speech dereverberation based on the WPE variants.

author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

import numpy as np
import soundfile as sf

import torch
torch.set_printoptions(precision=10)

from torch_custom.torch_utils import load_checkpoint, to_arr
from torch_custom.iterative_wpe import IterativeWPE
from torch_custom.neural_wpe import NeuralWPE

from bldnn_4M62 import LstmDnnNet as LPSEstimator
from gcunet4c_4M4390 import VACENet
from vace_wpe import VACEWPE


## ----------------------------------------------------- ##
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('device = {}'.format(device))
# device = "cpu"

stft_opts_torch = dict(
  n_fft=1024, hop_length=256, win_length=1024, win_type='hanning', 
  symmetric=True)
fft_bins = stft_opts_torch['n_fft']//2 + 1

# mfcc_opts_torch = dict(
#   fs=16000, nfft=1024, lowfreq=20., maxfreq=7600., 
#   nlinfilt=0, nlogfilt=40, nceps=40, 
#   lifter_type='sinusoidal', lift=-22.0) # only useful during fine-tuning

delay, taps1, taps2 = 3, 30, 15
## ----------------------------------------------------- ##


def apply_drc(audio, drc_ratio=0.25, n_pop=100, dtype='float32'):
  normalized = (max(audio.max(), abs(audio.min())) <= 1.0)
  normalizer = 1.0 if normalized else float(2**15)
  ## compute MMD
  audio_sorted = np.sort(audio.squeeze(), axis=-1)  # either (N,) or (D, N)
  audio_mmd = audio_sorted[..., -1:-n_pop-1:-1].mean(dtype=dtype) \
            - audio_sorted[..., :n_pop].mean(dtype=dtype)
  drange_gain = 2 * (normalizer/audio_mmd) * drc_ratio
  return (audio * drange_gain).astype(dtype), drange_gain.astype(dtype)


def run_vace_wpe(wpath, prefix, pretrain_opt='late', simp_opt='b'):
  assert pretrain_opt == 'late' # VACENet should be pretrained to estimate late reverberation components
  assert simp_opt == 'b' # Simplified VACE-WPE architecture is used
  print(f'Running "{prefix}-VACE-WPE"...')

  ## Saved checkpoint file
  prefix = prefix.lower()
  if prefix == 'drv':       # Drv-VACE-WPE
    ckpt_file = 'models/20210615-183131/ckpt-ep60'
  elif prefix == 'dns':     # Dns-VACE-WPE
    ckpt_file = 'models/20210615-221501/ckpt-ep60'
  elif prefix == 'dr-tsoc': # DR-TSO_\mathcal{C}-VACE-WPE
    ckpt_file = 'models/20210617-095331/ckpt-ep30'
  elif prefix == 'tsoc':    # TSO_\mathcal{C}-VACE-WPE
    ckpt_file = 'models/20210617-103601/ckpt-ep30'
  elif prefix == 'tson':    # TSO_\mathcal{N}-VACE-WPE
    ckpt_file = 'models/20210617-125831/ckpt-ep30'

  ## ------------------------------------------------- ##

  ## VACENet
  fake_scale = 2.0
  vacenet = VACENet(
    input_dim=fft_bins, stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='vace_unet', fake_scale=fake_scale)
  vacenet = vacenet.to(device)
  vacenet.eval()
  # print('VACENet size = {:.2f}M'.format(vacenet.size))
  # vacenet.check_trainable_parameters()

  ## LPSNet
  lpsnet = LPSEstimator(
    input_dim=fft_bins, 
    stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='ldnn_lpseir_ns')
  lpsnet = lpsnet.to(device)
  lpsnet.eval()
  # lpsnet.freeze() # should be frozen when fine-tuning the VACENet
  # print('LPSNet size = {:.2f}M'.format(lpsnet.size))
  # lpsnet.check_trainable_parameters()

  ## VACE-WPE
  dnn_vwpe = VACEWPE(
    stft_opts=stft_opts_torch, 
    lpsnet=lpsnet, vacenet=vacenet)#, 
    # mfcc_opts=mfcc_opts_torch) # only useful when fine-tuning the VACENet
  dnn_vwpe, *_ = load_checkpoint(dnn_vwpe, checkpoint=ckpt_file, strict=False)
  dnn_vwpe.to(device)
  dnn_vwpe.eval()
  # print('VACE-WPE size = {:.2f}M'.format(dnn_vwpe.size))
  # dnn_vwpe.check_trainable_parameters()

  ## ------------------------------------------------- ##

  ## Load audio and apply DRC
  aud, fs = sf.read(wpath) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, samples)
  with torch.no_grad():
    ## The input audio is in shape (batch, samples) (always assume #channels == 1)
    enh = dnn_vwpe.dereverb(
      aud.to(device), delay=delay, taps=taps2) # (t,)
    print(np.abs(enh).sum())
  ## Save
  output_wav_path = f'data/{prefix}-vace_wpe_taps{taps2}.wav'
  sf.write(output_wav_path, data=enh, samplerate=fs)


def run_neural_wpe(wpath, chs='single', dtype=torch.float64):
  print(f'Running "Neural-WPE-{chs}"...')

  ckpt_file = np.random.choice([
    'models/20210615-183131/ckpt-ep60', # Drv-VACE-WPE
    'models/20210615-221501/ckpt-ep60', # Dns-VACE-WPE
    'models/20210617-095331/ckpt-ep30', # TSON-VACE-WPE
    'models/20210617-103601/ckpt-ep30', # TSOC-VACE-WPE
    'models/20210617-125831/ckpt-ep30', # DR-TSOC-VACE-WPE
  ]) # the VACE-WPE variants share the same LPSNet model for PSD estimation

  ## ------------------------------------------------- ##

  ## LPSNet
  lpsnet = LPSEstimator(
    input_dim=fft_bins, 
    stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='ldnn_lpseir_ns')
  lpsnet = lpsnet.to(device)
  # lpsnet.freeze() # should be frozen when fine-tuning the VACENet
  lpsnet.eval()
  # print('LPSNet size = {:.2f}M'.format(lpsnet.size))
  # lpsnet.check_trainable_parameters()

  ## Neural WPE
  dnn_wpe = NeuralWPE(
    stft_opts=stft_opts_torch, 
    lpsnet=lpsnet)
  dnn_wpe, *_ = load_checkpoint(dnn_wpe, checkpoint=ckpt_file, strict=False)
  dnn_wpe.to(device)
  dnn_wpe.eval()
  # print('Neural WPE size = {:.2f}M'.format(dnn_wpe.size))
  # dnn_wpe.check_trainable_parameters()

  ## ------------------------------------------------- ##

  ## Load audio and apply DRC
  aud, fs = sf.read(wpath) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()

  if chs == 'single':
    aud = aud[None] # (channels=1, samples)
    taps = taps1
  if chs == 'dual':
    aud2, fs2 = sf.read(sample_wav2, dtype='float32')
    aud2 = aud2 * drc_gain
    aud = np.stack((aud, aud2), axis=0) # (channels=2, samples)
    taps = taps2

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, channels, samples)
  with torch.no_grad():
    ## The input audio must be in shape (batch, channels, samples)
    enh = dnn_wpe(
      aud.to(device), delay=delay, taps=taps, dtype=dtype) # (t,)
  enh = to_arr(enh).squeeze() # convert to numpy array and squeeze
  print(np.abs(enh).sum())
  ## Save
  if chs == 'dual':
    enh = enh[0] # only save the first channel
  # print(enh.sum())
  output_wav_path = f'data/nwpe_{chs}_taps{taps}.wav'
  sf.write(output_wav_path, data=enh, samplerate=fs)


def run_iterative_wpe(wpath, chs='single', n_iter=1, dtype=torch.float64):
  print(f'Running "Iterative-WPE-{chs}"...')

  ## IterativeWPE WPE
  iter_wpe = IterativeWPE(
    stft_opts=stft_opts_torch)

  ## Load audio and apply DRC
  aud, fs = sf.read(wpath) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()

  if chs == 'single':
    aud = aud[None] # (channels=1, samples)
    taps = taps1
  if chs == 'dual':
    aud2, fs2 = sf.read(sample_wav2, dtype='float32')
    aud2 = aud2 * drc_gain
    aud = np.stack((aud, aud2), axis=0) # (channels=2, samples)
    taps = taps2

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, channels, samples)
  with torch.no_grad():
    ## The input audio must be in shape (batch, channels, samples)
    enh = iter_wpe(
      aud.to(device), delay=delay, taps=taps, dtype=dtype) # (t,)
  enh = to_arr(enh).squeeze() # convert to numpy array and squeeze
  ## Save
  if chs == 'dual':
    enh = enh[0] # only save the first channel
  output_wav_path = f'data/iwpe_{chs}_taps{taps}_iter{n_iter}.wav'
  sf.write(output_wav_path, data=enh, samplerate=fs)



if __name__=="__main__":
  dtype = torch.float64

  sample_wav = 'data/AMI_WSJ20-Array1-1_T10c0201.wav'
  # sample_wav2 = 'data/AMI_WSJ20-Array1-2_T10c0201.wav'

  sample_wav = 'data/VOiCES_2019_Challenge_SID_eval_1327.wav' # babble noise
  # sample_wav = 'data/VOiCES_2019_Challenge_SID_eval_8058.wav' # ambient noise
  # sample_wav = 'data/VOiCES_2019_Challenge_SID_eval_11391.wav' # music + vocal



  ## Save DRC-applied raw signal
  aud, fs = sf.read(sample_wav) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()
  sample_wav_drc = 'data/raw_signal.wav'
  sf.write(sample_wav_drc, data=aud, samplerate=fs)

  # ## Iterative WPE
  # run_iterative_wpe('single', n_iter=1, dtype=dtype)
  # ### run_iterative_wpe('dual', n_iter=1, dtype=dtype)

  ## Neural WPE
  run_neural_wpe(sample_wav, 'single', dtype=dtype)
  ### run_neural_wpe('dual', dtype=dtype)

  ## VACE-WPE
  run_vace_wpe(sample_wav, prefix='drv')
  run_vace_wpe(sample_wav, prefix='dns')
  run_vace_wpe(sample_wav, prefix='tson')
  run_vace_wpe(sample_wav, prefix='tsoc')
  run_vace_wpe(sample_wav, prefix='dr-tsoc')
