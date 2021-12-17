"""
PyTorch implementation of batch-mode neural WPE.

This is basically a PyTorch translation of the NARA-WPE written in TensorFlow 
(https://github.com/fgnt/nara_wpe/blob/master/nara_wpe/tf_wpe.py), but with 
some modifications (e.g., rearrangement of some operations and zero-padding).

author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch
torch.set_printoptions(precision=10)
import torch.nn.functional as F

from torch_custom.signal_utils import signal_framing
from torch_custom.math_utils import complex_matrix_inverse, complex_matmul, complex_einsum


def wpe_mb_torch_ri(X, psd, taps=10, delay=3, ref_ch=None):
  """ X in shape ((B,) D, F, T(, 2))
      psd in shape ((B,) F, T)
  """
  ndim = X.dim()
  if ndim == 4: # a single utterance
    X, psd = X.unsqueeze(0), psd.unsqueeze(0) # batch dim

  Xr, Xi = X[..., 0], X[..., 1] # (B,D,F,T)
  B, D, Fb, T = Xr.size()

  Xr = Xr.permute(0,2,1,3).reshape(B*Fb, D, T) # (B,D,F,T) >> (B,F,D,T) >> (BF,D,T)
  Xi = Xi.permute(0,2,1,3).reshape(B*Fb, D, T) # (B,D,F,T) >> (B,F,D,T) >> (BF,D,T)
  psd = psd.view(B*Fb, T) # (B,F,T) >> (BF,T)
  Zr, Zi = wpe_torch_mf_ri(Xr, Xi, psd, taps, delay, ref_ch=ref_ch) # (BF,D,T)
  if ref_ch is not None: D = 1
  Zr = Zr.view(B, Fb, D, T).permute(0,2,1,3) # (BF,D,T) >> (B,F,D,T) >> (B,D,F,T)
  Zi = Zi.view(B, Fb, D, T).permute(0,2,1,3) # (BF,D,T) >> (B,F,D,T) >> (B,D,F,T)

  Z = torch.stack((Zr, Zi), dim=-1) # (B,D,F,T,2)
  if ndim == 4:
    return Z[0]
  return Z


def get_stacked_frames(X, delay, taps):
  """ Stack the past "taps" time frames with a "delay".
      Take X in (..., T) and returns X_tilde in (..., T, taps).
      (*Latest time frame comes first.)
  """
  ## Zero-pad the front side
  X_pad = F.pad(X, pad=(delay+taps-1, 0), mode='constant', value=0) # (B,D,F,T+dt1)
  ## Make frames of the padded input signal
  X_tilde = signal_framing(X_pad, frame_length=taps, frame_step=1, dim=-1)
  if delay > 0:
    X_tilde = X_tilde[..., :-delay, :] # (B,D,F,T,taps)
  return X_tilde.flip(dims=(-1,)) # (B,D,F,T,taps)

def wpe_torch_mf_ri(Xr, Xi, psd, taps=10, delay=3, ref_ch=None):
  """ X in shape (F, D, T)
      psd in shape (F, T)
  """
  # dt1 = delay + taps - 1

  Fb, D, T = Xr.size()
  Dtaps = D*taps

  ## 1) Get inverse of the power (simple reciprocal with flooring)
  eps = 1e-10 * psd.max(dim=-1, keepdim=True)[0] # (F,1)
  inv_pow = psd.max(eps).reciprocal_().unsqueeze(dim=-2) # (F,1,T)
  # inv_pow = psd.clamp_(eps).reciprocal_().unsqueeze(dim=-2) # (F,1,T)

  ## 2) Construct the matrix of the past "taps" samples for LP
  Xr_tilde = get_stacked_frames(Xr, delay=delay, taps=taps) # (F,D,T,taps)
  Xi_tilde = get_stacked_frames(Xi, delay=delay, taps=taps) # (F,D,T,taps)

  ## 3) Compute correlation matrix & correlation vector
  Xr_tilde_inv_pow = Xr_tilde * inv_pow[..., None] # (F,D,T,taps)
  Xi_tilde_inv_pow = Xi_tilde * inv_pow[..., None] # (F,D,T,taps)
  # print(Xr_tilde_inv_pow.abs().sum().item())

  """ R = X_tilde_inv_pow * hermite(X_tilde) """
  # corr_mat = torch.einsum('fdtk,fetl->fkdle', X_tilde_inv_pow, X_tilde.conj())
  corr_mat_r, corr_mat_i = complex_einsum('fdtk,fetl->fkdle', 
    Xr_tilde_inv_pow, Xi_tilde_inv_pow, Xr_tilde, Xi_tilde, 
    conjugate_b=True) # (F,taps,D,taps,D)
  corr_mat_r = torch.reshape(corr_mat_r, (Fb, Dtaps, Dtaps))
  corr_mat_i = torch.reshape(corr_mat_i, (Fb, Dtaps, Dtaps))

  """ P = X_tilde_inv_pow * hermite(X) """
  if ref_ch is not None:
    # D, X = 1, X[:,[ref_ch]]
    D, Xr, Xi = 1, Xr[:,[ref_ch]], Xi[:,[ref_ch]] # override
  # corr_vec = torch.einsum('fdtk,fet->fkde', X_tilde_inv_pow, X.conj())
  corr_vec_r, corr_vec_i = complex_einsum('fdtk,fet->fkde', 
    Xr_tilde_inv_pow, Xi_tilde_inv_pow, Xr, Xi, 
    conjugate_b=True) # (F,taps,D,D)
  corr_vec_r = torch.reshape(corr_vec_r, (Fb, Dtaps, D))
  corr_vec_i = torch.reshape(corr_vec_i, (Fb, Dtaps, D))

  ## 4) Compute LP filter coefficients
  """ G = (R^-1)P """
  # # lp_filter = torch.matmul(corr_mat.inverse(), corr_vec) # (D,Dtaps)
  # lp_filter = torch.solve(corr_mat, corr_vec) # (D,Dtaps)
  corr_imat_r, corr_imat_i = complex_matrix_inverse(
    corr_mat_r, corr_mat_i)
  filter_r, filter_i = complex_matmul(
    corr_imat_r, corr_imat_i, corr_vec_r, corr_vec_i) # (F,Dtaps,D)

  ## 5) Perform linear filtering for dereverberation
  Xr_tilde_rshp = Xr_tilde.permute(0,3,1,2).reshape(Fb, Dtaps, T) # (F,Dtaps,T)
  Xi_tilde_rshp = Xi_tilde.permute(0,3,1,2).reshape(Fb, Dtaps, T) # (F,Dtaps,T)
  # X_tail = lp_filter.conj().transpose(1, 2).matmul(X_tilde_rshp)
  Xr_tail, Xi_tail = complex_matmul(
    filter_r, filter_i, Xr_tilde_rshp, Xi_tilde_rshp, 
    hermite_a=True) # (F,D,T)
  Zr, Zi = Xr - Xr_tail, Xi - Xi_tail # (F,D,T)
  return Zr, Zi

