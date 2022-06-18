"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_custom.torch_utils import get_l2_norm


def update_dict(dic1, dic2):
  for key in dic2:
    assert key not in dic1
  dic1.update(dic2)
  del dic2


def get_nonlinearity(name, inplace=False, kwargs={}):
  if name is None:
    return nn.Identity()
  else:
    assert isinstance(name, str)
    name = name.lower()
    if name == 'elu':
      return nn.ELU(inplace=inplace)
    elif name == 'selu':
      return nn.SELU(inplace=inplace)
    elif name == 'celu':
      return nn.CELU(inplace=inplace, **kwargs) # alpha=1.0
    elif name == 'gelu':
      return nn.GELU()
    elif name == 'silu' or name == 'swish':
      return nn.SiLU(inplace=inplace)

    elif name == 'relu':
      return nn.ReLU(inplace=inplace)
    elif name == 'relu6':
      return nn.ReLU6(inplace=inplace)
    elif name == 'lrelu':
      return nn.LeakyReLU(inplace=inplace, **kwargs) # negative_slope=0.01
    elif name == 'prelu':
      return nn.PReLU(**kwargs)
    elif name == 'rrelu':
      return nn.RReLU(inplace=inplace, **kwargs) # lower=0.125, upper=0.3333333333333333

    elif name == 'tanh':
      return nn.Tanh()
    elif name == 'sigmoid':
      return nn.Sigmoid()
    elif name == 'softsign':
      return nn.Softsign()
    elif name == 'softplus':
      return nn.Softplus(**kwargs) # beta=1, threshold=20


class CustomModel(nn.Module):
  def __init__(self, *args, **kwargs):
    super(CustomModel, self).__init__(*args, **kwargs)

  def trainable_parameters(self):
    return (param for param in self.parameters() if param.requires_grad)

  # @property
  # def weights_l2_norm(self):
  #   return sum([get_l2_norm(w) for w in self.weights])

  def weights_l2_norm(self):
    l2_norms = [get_l2_norm(w) for w in self.weights_list]
    l2_norm_sum = sum(l2_norms)
    l2_norm_dict = {key:val for key,val in zip(self.weights_name, l2_norms)}
    return l2_norm_sum, l2_norm_dict

  @property
  def size(self):
    return sum(p.numel() for p in self.parameters()) / float(1e+6)

  @property
  def size_trainable(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad) / float(1e+6)

  @property
  def device(self):
    return next(self.parameters()).device

  def freeze(self):
    for param in self.parameters():
      param.requires_grad_(False)

  def unfreeze(self):
    for param in self.parameters():
      param.requires_grad_(True)

  def freeze_named(self, exclude=[]):
    if isinstance(exclude, str): exclude = [exclude]
    for name, param in self.named_parameters():
      will_exclude = False
      for name_to_exclude in exclude:
        # if name.startswith(name_to_exclude):
        if re.search(name_to_exclude, name):
          will_exclude = True

      if not will_exclude:
        param.requires_grad_(False)
        print('[Freezed] %s' % name)

  def change_mode(self, mode, attr_list):
    assert mode in [True, False]
    assert isinstance(attr_list, list) and len(attr_list)
    for attr in attr_list:
      getattr(self, attr).train(mode)
      print('Set {}.train({})'.format(attr, mode))

  def check_trainable_parameters(self):
    print('\n===============  Check Parameters Status  ===============')
    for name, param in self.named_parameters():
      print(name, param.requires_grad)
    print('=========================================================\n')



class GlobalMeanVarianceNormalizer(nn.Module):
  def __init__(self, mean, std):
    super(GlobalMeanVarianceNormalizer, self).__init__()
    self.register_buffer('mean', torch.from_numpy(mean))
    self.register_buffer('std', torch.from_numpy(std))

  def forward(self, x):
    return (x - self.mean.to(x.device)) / self.std.to(x.device)

  def denormalize(self, x):
    return x*self.std.to(x.device) + self.mean.to(x.device)

class SlidingWindowMeanNorm(nn.Module):
  def __init__(self, swLen, padtype="reflect"):
    """ Input  must be (N, C, W_in) 
        Output must be (N, C, W_in_padded)
        (The last dim is assumed to be the time-frame axis)
    """
    super(SlidingWindowMeanNorm, self).__init__()
    padLen = int(swLen/2)
    self.padder = nn.ReflectionPad1d((padLen, padLen-1))
    self.swLen = swLen
    self.time_axis = -1

  def forward(self, x):
    if x.size(self.time_axis) > self.swLen:
      swMean = F.avg_pool1d(
        self.padder(x), kernel_size=self.swLen, stride=1)
    else:
      swMean = torch.mean(x, dim=self.time_axis, keepdim=True)
    return x - swMean



class Glu2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='Glu2d'):
    super(Glu2d, self).__init__()
    self.conv = nn.Conv2d(
      in_channels, 2*out_channels, kernel_size, stride, padding, 
      dilation, groups, bias, padding_mode)
    self.weights = {scope+'_kernel':self.conv.weight}

  def forward(self, x, drop=0.0):
    x = F.glu(self.conv(x), dim=1)
    if drop:
      x = F.dropout(x, p=drop, training=self.training, inplace=True)
    return x


class Conv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='Conv2d'):
    super(Conv2d, self).__init__()
    self.conv = nn.Conv2d(
      in_channels, out_channels, kernel_size, stride, padding, 
      dilation, groups, bias, padding_mode)
    self.weights = {scope+'_kernel':self.conv.weight}

  def forward(self, x, drop=0.0):
    x = self.conv(x)
    if drop:
      x = F.dropout(x, p=drop, training=self.training, inplace=True)
    return x

class Conv2dTp(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               out_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', 
               scope='Conv2dTp'):
    super(Conv2dTp, self).__init__()
    self.convtp = nn.ConvTranspose2d(
      in_channels, out_channels, kernel_size, stride, padding, 
      output_padding=out_padding, groups=groups, bias=bias, dilation=dilation, 
      padding_mode=padding_mode)
    self.weights = {scope+'_kernel':self.convtp.weight}

  def forward(self, x, output_size=None, drop=0.0):
    x = self.convtp(x, output_size=output_size)
    if drop:
      x = F.dropout(x, p=drop, training=self.training, inplace=True)
    return x


class Conv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
               dilation=1, groups=1, bias=True, padding_mode='zeros', 
               scope='Conv1d'):
    super(Conv1d, self).__init__()
    self.conv = nn.Conv1d(
      in_channels, out_channels, kernel_size, stride, padding, 
      dilation, groups, bias, padding_mode)
    self.weights = {scope+'_kernel':self.conv.weight}

  def forward(self, x, drop=0.0):
    x = self.conv(x)
    if drop:
      x = F.dropout(x, p=drop, training=self.training, inplace=True)
    return x


class Dense(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, bias: bool = True, scope: str = 'Dense'):
    super(Dense, self).__init__()
    self.dense = nn.Linear(in_dim, out_dim, bias)
    self.weights = {scope+'_kernel':self.dense.weight}

  def forward(self, x, drop=0.0):
    x = self.dense(x)
    if drop:
      x = F.dropout(x, p=drop, training=self.training, inplace=True)
    return x

