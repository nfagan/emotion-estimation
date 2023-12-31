import numpy as np
import h5py
import scipy.io
from typing import List, Union, Tuple
from sklearn.decomposition import PCA
import joblib

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

def h5_create_datasets(f, act_shape: Tuple[int]):
  nr = act_shape[0]
  f.create_dataset('activations', act_shape, 'float32')
  f.create_dataset('splits', (nr, 1), 'S10')
  f.create_dataset('layers', (nr, 1), 'S10')
  f.create_dataset('identifiers', (nr, 1), 'S10')

def h5_write_into_datasets(
    f, activations: np.array, 
    splits: Union[List[str], str], 
    layers: Union[List[str], str],
    identifiers: Union[List[str], str],
    offset: int):
  """
  activations: (image x dimension1 x ...)
  splits: (image,)            'train' or 'valid', for each image
  layers: (image,)            layer names
  identifiers: (image,)       image identifiers (e.g. filenames)
  offset:                     row offset
  """
  def maybe_list(x):
    return list(x) if not isinstance(x, list) else x
  def to_str_list(x):
    return [x] * activations.shape[0] if isinstance(x, str) else maybe_list(x)
    
  splits = to_str_list(splits)
  layers = to_str_list(layers)
  identifiers = to_str_list(identifiers)

  nr = activations.shape[0]
  f['activations'][offset:offset+nr, :] = activations
  f['splits'][offset:offset+nr, 0] = splits
  f['layers'][offset:offset+nr, 0] = layers
  f['identifiers'][offset:offset+nr, 0] = identifiers

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

def load_pca_checkpoint(src_p: str) -> PCA:
  mat = joblib.load(src_p)
  model = mat['model']
  return model, mat['hp'] if 'hp' in mat else None

def save_pca_checkpoint(save_p: str, model: PCA):
  sd = {'model': model}
  joblib.dump(sd, save_p)