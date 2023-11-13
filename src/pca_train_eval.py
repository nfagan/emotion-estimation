import utility
import numpy as np
import scipy.io
import h5py
import os
import joblib
from sklearn.decomposition import PCA

data_root = utility.get_dataroot()

MODE = 'valid'
SELECT_SPLIT = 'valid'
SAVE = True
CUSTOM_NUM_COMPONENTS = 80
# CUSTOM_NUM_COMPONENTS = None

DATASET = 'd3dfr'
VARIANT = '_expression_balanced_var_subsample_1000'
LAYERS = [f'resnet_layer{x+1}' for x in range(4)] + ['resnet_output', 'ReconNetWrapper_output']
LAYERS = LAYERS[1:]
LAYERS = ['resnet_layer1']

# DATASET = 'resnet_image_embedding'
# VARIANT = '_expression_balanced_var_subsample_1000'
# LAYERS = [f'layer{x+2}' for x in range(3)]
# LAYERS = ['layer1']

def make_model_from_hp(hp, nc=None):
  return PCA(n_components=nc)

def train_hp(act):
  hp = {}
  hp['var_explained_thresh'] = 0.9
  return hp

def load_checkpoint(src_p: str) -> PCA:
  mat = joblib.load(os.path.join(src_p, 'cp.pkl'))
  model = mat['model']
  return model, mat['hp']

def save_checkpoint(save_p, model, hp, error, ev, batch_error):
  os.makedirs(save_p, exist_ok=True)
  sd = {
    'components': model.components_,
    'mean': model.mean_,
    'hp': hp,
    'error': error,
    'batch_error': batch_error,
    'frac_explained_variance': ev
  }
  # save as matlab, numpy, and pickle
  scipy.io.savemat(os.path.join(save_p, 'cp.mat'), sd)
  np.save(os.path.join(save_p, 'cp.npy'), sd)
  # only pickle model
  sd['model'] = model
  joblib.dump(sd, os.path.join(save_p, 'cp.pkl'))

def recon_err(model, act, nc):
  act_trans = model.transform(act)
  act_trans[:, nc:] = 0.
  act_proj = model.inverse_transform(act_trans)
  return np.sum((act - act_proj) ** 2, axis=1)

def explained_var(model, act, nc):
  act_trans = model.transform(act)
  act_trans[:, nc:] = 0.
  act_proj = model.inverse_transform(act_trans)
  return np.mean(np.var(act_proj, axis=0)) / np.mean(np.var(act, axis=0))
  
if __name__ == '__main__':
  for li in range(len(LAYERS)):
    print(f'{LAYERS[li]} ({li+1} of {len(LAYERS)})')
    
    layer = LAYERS[li]
    acts = f'{DATASET}/{MODE}{VARIANT}/{layer}'
    pca_p = f'pca_nc_{CUSTOM_NUM_COMPONENTS}' if CUSTOM_NUM_COMPONENTS is not None else 'pca'
    cp_dir = f'{pca_p}_checkpoints' if MODE == 'train' else f'{pca_p}_eval'
    dst_p = os.path.join(data_root, cp_dir, acts)

    act, split, _, _ = utility.load_activations(os.path.join(data_root, 'activations', f'{acts}.h5'))
    act = act[split == SELECT_SPLIT, :]

    if MODE == 'train':
      hp = train_hp(act)
      model = make_model_from_hp(hp)
      model.fit(act)

      # determine number of components
      ve = np.cumsum(model.explained_variance_)
      ve = ve / ve[-1]
      if CUSTOM_NUM_COMPONENTS is None:
        nc = 1 + np.argwhere(ve > hp['var_explained_thresh'])[0][0]
      else:
        nc = CUSTOM_NUM_COMPONENTS
      hp['num_components'] = nc

      err = recon_err(model, act, nc)
      ev = explained_var(model, act, nc)

      if SAVE:    
        save_checkpoint(dst_p, model, hp, err, ev, np.mean(err))
    
    else:
      cp_dir_name = f'{pca_p}_checkpoints'
      src_p = os.path.join(data_root, cp_dir_name, acts.replace('valid', 'train'))
      model, hp = load_checkpoint(src_p)
      err = recon_err(model, act, hp['num_components'])
      ev = explained_var(model, act, hp['num_components'])
      
      if SAVE:
        save_checkpoint(dst_p, model, hp, err, ev, np.mean(err))