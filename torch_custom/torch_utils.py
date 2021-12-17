import os, re
import torch

def shape(tensor):
  return tuple(tensor.size())

def to_gpu(x, device=None):
  if not isinstance(x, torch.Tensor):
    x = torch.from_numpy(x)
  return x.contiguous().cuda(device=device, non_blocking=True)

def to_arr(x):
  if isinstance(x, (list, tuple)):
    return tuple([to_arr(_x) for _x in x])
  else:
    return x.cpu().detach().numpy()


def get_l2_norm(w):
  return 0.5 * (w**2).sum() # numpy-compatible command
  return 0.5 * torch.sum(w**2)

def l2_normalize(tensor, dim, eps=1e-12, return_norm=False):
  l2_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True).clamp(eps)
  if return_norm:
    return tensor / l2_norm, l2_norm
  else:
    return tensor / l2_norm

def l2_norm(tensor, dim, eps=1e-12):
  l2_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
  l2_norm = torch.max(l2_norm, eps*torch.ones_like(l2_norm))
  return l2_norm


def set_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def load_checkpoint(model, optimizer=None, checkpoint='dummy_path', strict=True):
  assert os.path.isfile(checkpoint), '%s does not exist!' % checkpoint
  ckpt_dict = torch.load(checkpoint, map_location='cpu')
  model.load_state_dict(ckpt_dict['state_dict'], strict=strict)
  if optimizer is not None:
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    for param_group in optimizer.param_groups:
      param_group['lr'] = ckpt_dict['learning_rate']
  # print('Loaded checkpoint %s from iteration %d' % 
  #     (checkpoint, ckpt_dict['iteration']))
  return model, ckpt_dict['learning_rate'], ckpt_dict['iteration']

def load_model_state_dict(model, checkpoint, exclude=[], strict=True):
  assert os.path.isfile(checkpoint), '%s does not exist!' % checkpoint
  ckpt_dict = torch.load(checkpoint, map_location='cpu')

  if isinstance(exclude, str): exclude = [exclude]
  if len(exclude):
    strict = False
    num_keys_before = len(ckpt_dict['state_dict'])
    for key_to_load in ckpt_dict['state_dict']:
      for key_to_exclude in exclude:
        if re.search(key_to_exclude, key_to_load):
          ckpt_dict['state_dict'].pop(key_to_load)
    num_keys_after = len(ckpt_dict['state_dict'])

    if num_keys_before > num_keys_after:
      info = 'Excluding {}... >> '.format(exclude)
      info += 'From %d, excluded %d, left %d keys...' % (
        num_keys_before, num_keys_before-num_keys_after, num_keys_after)
      print(info)
    else:
      raise ValueError('Tried to exclude {}, but failed!'.format(exclude))

  model.load_state_dict(ckpt_dict['state_dict'], strict=strict)
  return model

def save_checkpoint(model, optimizer, iteration, learning_rate, ckpt_file):
  torch.save({
    'iteration': iteration, 
    'state_dict': model.state_dict(), 
    'optimizer': optimizer.state_dict(), 
    'learning_rate': learning_rate}, 
    ckpt_file)
