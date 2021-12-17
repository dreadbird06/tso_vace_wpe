"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch


def complex_matrix_inverse_reg(mat_r, mat_i, reg=1e-3):
  """ M = A + iB 
      Re{M^-1} = (A + B(A^-1)B)^-1
      Im{M^-1} = -Re{M^-1} * B(A^-1)
      (https://kr.mathworks.com/matlabcentral/fileexchange/49373-complex-matrix-inversion-by-real-matrix-inversion)
  """
  try:
    dtype = mat_r.dtype
    ## --------------------------------------- ##
    mat_r, mat_i = mat_r.double(), mat_i.double()
    ## --------------------------------------- ##
    mat_r_inv = mat_r.inverse()
    tmp_mat = mat_i.matmul(mat_r_inv)
    imat_r = torch.inverse(mat_r + tmp_mat.matmul(mat_i))
    imat_i = -imat_r.matmul(tmp_mat)
    # return imat_r.float(), imat_i.float()
    return imat_r.type(dtype), imat_i.type(dtype)

  except:
    """ Tikhonov regularization
        (https://gitlab.uni-oldenburg.de/hura4843/deep-mfmvdr/-/blob/master/deep_mfmvdr/utils.py)
    """
    ## ------------------------------------- ##
    mat_r, mat_i = mat_r.float(), mat_i.float()
    ## ------------------------------------- ##
    nrows = mat_r.size(-2)
    eye = torch.eye(nrows, dtype=mat_r.dtype, device=mat_r.device)
    eye_shape = tuple([1]*(mat_r.dim()-2)) + (nrows, nrows)
    eye = eye.view(*eye_shape)

    mat_abs = torch.sqrt(mat_r.square() + mat_i.square())
    trace = torch.sum(eye*mat_abs, dim=(-2,-1), keepdim=True)
    scale = reg / nrows * trace
    scale = scale.detach() # treated as a constant when running backprop
    return complex_matrix_inverse(mat_r+scale, mat_i)

def complex_matrix_inverse(mat_r, mat_i):
  """ M = A + iB 
      Re{M^-1} = (A + B(A^-1)B)^-1
      Im{M^-1} = -Re{M^-1} * B(A^-1)
      (https://kr.mathworks.com/matlabcentral/fileexchange/49373-complex-matrix-inversion-by-real-matrix-inversion)
  """
  dtype = mat_r.dtype
  ## --------------------------------------- ##
  mat_r, mat_i = mat_r.double(), mat_i.double()
  ## --------------------------------------- ##
  try:
    mat_r_inv = mat_r.inverse()
  except:
    mat_r_inv = mat_r.pinverse()
  tmp_mat = mat_i.matmul(mat_r_inv)
  try:
    imat_r = torch.inverse(mat_r + tmp_mat.matmul(mat_i))
  except:
    imat_r = torch.pinverse(mat_r + tmp_mat.matmul(mat_i))
  imat_i = -imat_r.matmul(tmp_mat)
  # return imat_r.float(), imat_i.float()
  return imat_r.type(dtype), imat_i.type(dtype)

def complex_mul(Ar, Ai, Br, Bi, conjugate_a=False, conjugate_b=False):
  """ C = A * B """
  if conjugate_a:
    Ai = Ai.neg()
  if conjugate_b:
    Bi = Bi.neg()
  Cr = Ar.mul(Br) - Ai.mul(Bi)
  Ci = Ai.mul(Br) + Ar.mul(Bi)
  return Cr, Ci

def complex_matmul(Ar, Ai, Br, Bi, hermite_a=False, hermite_b=False):
  """ C = A.matmul(B) """
  if hermite_a:
    Ar = Ar.transpose(-2, -1)
    Ai = Ai.transpose(-2, -1).neg()
  if hermite_b:
    Br = Br.transpose(-2, -1)
    Bi = Bi.transpose(-2, -1).neg()
  Cr = Ar.matmul(Br) - Ai.matmul(Bi)
  Ci = Ai.matmul(Br) + Ar.matmul(Bi)
  return Cr, Ci
  # return Ar.matmul(Br) - Ai.matmul(Bi), Ai.matmul(Br) + Ar.matmul(Bi)

def complex_einsum(equation, Ar, Ai, Br, Bi, conjugate_a=False, conjugate_b=False):
  """ C = torch.einsum(equation, A, B) """
  if conjugate_a:
    # Ar = Ar
    Ai = Ai.neg()
  if conjugate_b:
    # Br = Br
    Bi = Bi.neg()
  Cr = torch.einsum(equation, Ar, Br) - torch.einsum(equation, Ai, Bi)
  Ci = torch.einsum(equation, Ai, Br) + torch.einsum(equation, Ar, Bi)
  return Cr, Ci

def complex_tensor_normalize(tensor_r, tensor_i, denom=None, eps=1e-8):
  if denom is None:
    denom = (tensor_r.square() + tensor_i.square()).clamp_(eps).sqrt()
  return tensor_r.div_(denom), tensor_i.div_(denom)

def matrix_trace(matrix, eps=1e-8):
  return torch.einsum('...ii->...', matrix).clamp_(eps)

def complex_norm_l2(xr, xi, dim, keepdim=False, eps=1e-10):
  """ a^{H} * a """
  return (xr.square() + xi.square()).sum(
    dim=dim, keepdim=keepdim).clamp_(eps).sqrt()
  # return (xr.square() + xi.square()).clamp_(eps).sqrt()

