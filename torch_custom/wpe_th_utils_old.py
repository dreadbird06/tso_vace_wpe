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
from torch_custom.math_utils import complex_matrix_inverse


def wpe_mb_torch_ri(X, psd, taps=10, delay=3):
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
  Zr, Zi = wpe_torch_mf_ri(Xr, Xi, psd, taps, delay) # (BF,D,T)
  Zr = Zr.view(B, Fb, D, T).permute(0,2,1,3) # (BF,D,T) >> (B,F,D,T) >> (B,D,F,T)
  Zi = Zi.view(B, Fb, D, T).permute(0,2,1,3) # (BF,D,T) >> (B,F,D,T) >> (B,D,F,T)

  Z = torch.stack((Zr, Zi), dim=-1) # (B,D,F,T,2)
  if ndim == 4:
    return Z[0]
  return Z


def wpe_torch_mf_ri(Xr, Xi, psd, taps=10, delay=3):
  """ X in shape (F, D, T)
      psd in shape (F, T)
  """
  dt1 = delay + taps - 1

  Fb, D, T = Xr.size()
  Dtaps = D*taps

  ## ========================================== ##
  ## 1) Get inverse of the PSD (after flooring) ##
  ## ========================================== ##
  eps = 1e-10 * psd.max(dim=-1, keepdim=True)[0] # time-frame axis
  inv_power = psd.max(eps).reciprocal_().unsqueeze(dim=-2) # channel axis

  ## ========================================================= ##
  ## 2) Construct the matrix of the past "taps" samples for LP ##
  ## ========================================================= ##
  ## 2-1) Zero-pad the front-side of the signal
  Xr_pad = F.pad(Xr, pad=(dt1, 0), mode='constant', value=0)
  Xi_pad = F.pad(Xi, pad=(dt1, 0), mode='constant', value=0)

  ## 2-2) Framing to construct the matrix of past "taps" samples
  """ shape:  (F, D, T+dt1) >> (F, D, T+delay, taps) >> (F, D, T, taps) """
  Xr_tilde = signal_framing(Xr_pad, frame_length=taps, frame_step=1, 
    dim=-1)[..., :-delay, :].flip(dims=(-1,))
  Xi_tilde = signal_framing(Xi_pad, frame_length=taps, frame_step=1, 
    dim=-1)[..., :-delay, :].flip(dims=(-1,))

  ## ================================================== ##
  ## 3) Compute correlation matrix & correlation vector ##
  ## ================================================== ##
  Xr_tilde_inv_power = Xr_tilde * inv_power[..., None] # (F, D, T, taps)
  Xi_tilde_inv_power = Xi_tilde * inv_power[..., None] # (F, D, T, taps)

  """ corr_mat = X_tilde_inv_power * hermite(X_tilde) """
  # corr_mat = torch.einsum('fdtk,fetl->fkdle', X_tilde_inv_power, X_tilde.conj())
  corr_mat_r = torch.einsum('fdtk,fetl->fkdle', Xr_tilde_inv_power, Xr_tilde) \
             + torch.einsum('fdtk,fetl->fkdle', Xi_tilde_inv_power, Xi_tilde)
  corr_mat_i = torch.einsum('fdtk,fetl->fkdle', Xi_tilde_inv_power, Xr_tilde) \
             - torch.einsum('fdtk,fetl->fkdle', Xr_tilde_inv_power, Xi_tilde)
  corr_mat_r = torch.reshape(corr_mat_r, (Fb, Dtaps, Dtaps))
  corr_mat_i = torch.reshape(corr_mat_i, (Fb, Dtaps, Dtaps))

  """ corr_vec = X_tilde_inv_power * hermite(X) """
  # corr_vec = torch.einsum('fdtk,fet->fkde', X_tilde_inv_power, X.conj())
  corr_vec_r = torch.einsum('fdtk,fet->fkde', Xr_tilde_inv_power, Xr) \
             + torch.einsum('fdtk,fet->fkde', Xi_tilde_inv_power, Xi)
  corr_vec_i = torch.einsum('fdtk,fet->fkde', Xi_tilde_inv_power, Xr) \
             - torch.einsum('fdtk,fet->fkde', Xr_tilde_inv_power, Xi)
  corr_vec_r = torch.reshape(corr_vec_r, (Fb, Dtaps, D))
  corr_vec_i = torch.reshape(corr_vec_i, (Fb, Dtaps, D))

  ## ================================= ##
  ## 4) Compute LP filter coefficients ##
  ## ================================= ##
  """ G = (R^-1)P """
  # # lp_filter = torch.matmul(corr_mat.inverse(), corr_vec) # (D, Dtaps)
  # lp_filter = torch.solve(corr_mat, corr_vec) # (D, Dtaps)
  inv_corr_mat_r, inv_corr_mat_i = complex_matrix_inverse(corr_mat_r, corr_mat_i)
  lp_filter_r = torch.matmul(inv_corr_mat_r, corr_vec_r) \
              - torch.matmul(inv_corr_mat_i, corr_vec_i) # (F, Dtaps, D)
  lp_filter_i = torch.matmul(inv_corr_mat_i, corr_vec_r) \
              + torch.matmul(inv_corr_mat_r, corr_vec_i) # (F, Dtaps, D)

  ## =============================================== ##
  ## 5) Perform linear filtering for dereverberation ##
  ## =============================================== ##
  Xr_tilde_rshp = Xr_tilde.permute(0,3,1,2).reshape(Fb, Dtaps, T) # (F, Dtaps, T)
  Xi_tilde_rshp = Xi_tilde.permute(0,3,1,2).reshape(Fb, Dtaps, T) # (F, Dtaps, T)

  # X_tail = lp_filter.conj().transpose(1, 2).matmul(X_tilde_rshp)
  Xr_tail = torch.matmul(lp_filter_r.transpose(1, 2), Xr_tilde_rshp) \
          + torch.matmul(lp_filter_i.transpose(1, 2), Xi_tilde_rshp) # (F, D, T)
  Xi_tail = torch.matmul(lp_filter_r.transpose(1, 2), Xi_tilde_rshp) \
          - torch.matmul(lp_filter_i.transpose(1, 2), Xr_tilde_rshp) # (F, D, T)

  Zr, Zi = Xr - Xr_tail, Xi - Xi_tail # (F, D, T)
  return Zr, Zi
