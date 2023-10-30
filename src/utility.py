import numpy as np
import h5py
import scipy.io
from typing import List, Union

def get_dataroot():
  return 'D:\\data\\changlab\\ilker_collab'

def read_ascii(a):
  return np.array([str(x[0], 'ascii') for x in a])

def load_goldenberg_image_info(p):
  mfile = scipy.io.loadmat(p)['im_info_struct'][0, 0]
  idents = [x[0][0] for x in mfile[0]]
  ratings = [x[0] for x in mfile[1]]
  valences = [x[0][0] for x in mfile[2]]
  subjects = [x[0][0] for x in mfile[3]]
  return {'identifier': idents, 'rating': ratings, 'valence': valences, 'subject': subjects}

def load_activations(p):
  if isinstance(p, h5py.File):
    f = p
  else:
    f = h5py.File(p, 'r')
  acts = f['/activations'][:]
  splits = read_ascii(f['/splits'][:])
  fnames = read_ascii(f['/identifiers'][:])
  layers = read_ascii(f['/layers'][:])
  return acts, splits, fnames, layers

def h5_write_activations(
    f, activations: np.array, 
    splits: Union[List[str], str], 
    layers: Union[List[str], str],
    identifiers: Union[List[str], str]):
  """
  activations: (image x dimension1 x ...)
  splits: (image,)            'train' or 'valid', for each image
  layers: (image,)            layer names
  identifiers: (image,)       image identifiers (e.g. filenames)
  """
  def maybe_list(x):
    return list(x) if not isinstance(x, list) else x
  def to_str_list(x):
    return [x] * activations.shape[0] if isinstance(x, str) else maybe_list(x)
    
  splits = to_str_list(splits)
  layers = to_str_list(layers)
  identifiers = to_str_list(identifiers)

  f.create_dataset('activations', data=activations)
  f.create_dataset('splits', (len(splits), 1), 'S10', data=splits)
  f.create_dataset('layers', (len(layers), 1), 'S10', data=layers)
  f.create_dataset('identifiers', (len(identifiers), 1), 'S10', data=identifiers)