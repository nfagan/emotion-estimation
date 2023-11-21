import sparse_coding_model
import utility
import numpy as np
import scipy.io
import os
from sklearn.decomposition import PCA

data_root = utility.get_dataroot()

N_PCS = None
SEPARATE_TRAIN_VALID_PCA = False

MODE = 'train'
SELECT_SPLIT = 'train'
SAVE = True

DATASET = 'd3dfr'
VARIANT = '_var_subsample_2048'
VARIANT = '_expression_balanced_var_subsample_1000'
# VARIANT = ''
# LAYERS = [f'resnet_layer{x+1}' for x in range(4)] + ['resnet_output', 'ReconNetWrapper_output']
# LAYERS = [f'resnet_layer{x+1}' for x in range(4)]
LAYERS = [f'ReconNetWrapper_output_identity_expression']
# LAYERS = LAYERS[1:]
# LAYERS = ['resnet_layer1']

# DATASET = 'resnet_image_embedding'
# VARIANT = '_expression_balanced_var_subsample_1000'
# LAYERS = [f'layer{x+2}' for x in range(3)]
# LAYERS = ['layer1']

RESCALE_MEAN_ACTIVATIONS_TO = 0.5

# DATASET = 'image_net'
# VARIANT = ''
# LAYERS = [f'block{i+1}_pool' for i in range(5)]
# RESCALE_MEAN_ACTIVATIONS_TO = None

def make_model_from_hp(hp):
  return sparse_coding_model.OlshausenFieldModel(
      num_inputs=hp['codeword_dim'], 
      num_units=hp['num_units'], 
      batch_size=hp['batch_size'], 
      Phi=None, 
      lr_r=hp['lr_r'], 
      lr_Phi=hp['lr_Phi'], 
      lmda=hp['lmda'])

def load_checkpoint(src_p: str) -> sparse_coding_model.OlshausenFieldModel:
  mat = np.load(os.path.join(src_p, 'cp.npy'), allow_pickle=True).item()
  model = make_model_from_hp(mat['hp'])
  model.load_dict(mat['state_dict'])
  return model, mat['hp']

def save_checkpoint(save_p, model, hp, error, batch_error, prediction=None, pca_model=None):
  os.makedirs(save_p, exist_ok=True)
  sd = {
    'state_dict': model.to_dict(),
    'hp': hp,
    'error': error,
    'batch_error': batch_error
  }
  if prediction is not None:
    sd['prediction'] = prediction
  scipy.io.savemat(os.path.join(save_p, 'cp.mat'), sd)
  np.save(os.path.join(save_p, 'cp.npy'), sd)
  if pca_model is not None:
    utility.save_pca_checkpoint(os.path.join(save_p, 'cp.pkl'), pca_model)

def train_hp(act, pca_model):
  """
  see compute_pairwise_spc_distance_pca
  """
  lr = 1e-2
  # lr = 1e-3 # @NOTE: this is lower than Qi's, which was 1e-2
  hp = {}
  hp['codeword_dim'] = act.shape[1]
  if pca_model is None:
    hp['num_codewords'] = hp['num_units'] = 500
  else:
    hp['num_codewords'] = hp['num_units'] = pca_model.n_components // 2
  # hp['batch_size'] = 250
  hp['batch_size'] = 64 # @NOTE: this is lower than Qi's, which was 250
  hp['lr_r'] = lr
  hp['lr_Phi'] = lr
  hp['lmda'] = 5e-3 # @NOTE: Use for full resnet_output and ReconNetWrapper_output
  # hp['lmda'] = 5e-4 # @NOTE: this is lower than Qi's, which was 5e-3; use for var subsample 1000
  hp['lmda'] = 5e-3 # @NOTE: use for 2048; resnet_output fails to converge
  hp['eps'] = 1e-2
  # hp['eps'] = 1e-3  # @NOTE: this is lower than Qi's, which was 1e-2
  # hp['num_iter'] = 1000
  hp['num_iter'] = 250 # @NOTE: this is lower than Qi's, which was 1000
  hp['nt_max'] = 1000
  hp['verbose'] = False
  return hp
  
if __name__ == '__main__':
  for layer in LAYERS:
    src_ln = layer
    dst_ln = layer

    act_mask = None
    if dst_ln == 'ReconNetWrapper_output_identity_expression':
      src_ln = 'ReconNetWrapper_output'
      act_mask = np.arange(0, 80 + 64)  # alpha (R^80) = identity, beta (R^64) = expression

    src_acts = f'{DATASET}/{MODE}{VARIANT}/{src_ln}'
    dst_acts = f'{DATASET}/{MODE}{VARIANT}/{dst_ln}'

    sc_dir = 'sc' if N_PCS is None else f'sc_n_pc_{N_PCS}'
    train_cp_dir = f'{sc_dir}_checkpoints'
    cp_dir = train_cp_dir if MODE == 'train' else f'{sc_dir}_eval'

    src_p = os.path.join(data_root, train_cp_dir, dst_acts.replace('valid', 'train')) # load train checkpoint
    dst_p = os.path.join(data_root, cp_dir, dst_acts)

    act, split, _, _ = utility.load_activations(os.path.join(data_root, 'activations', f'{src_acts}.h5'))
    act = act[split == SELECT_SPLIT, :]

    if act_mask is not None:
      act = act[:, act_mask]

    pca_model = None
    if N_PCS is not None:
      if MODE == 'train' or SEPARATE_TRAIN_VALID_PCA:
        pca_model = PCA(n_components=N_PCS)
        act = pca_model.fit_transform(act)
      else:
        pca_model, _ = utility.load_pca_checkpoint(os.path.join(src_p, 'cp.pkl'))
        act = pca_model.transform(act)

    if RESCALE_MEAN_ACTIVATIONS_TO is not None:
      mult = RESCALE_MEAN_ACTIVATIONS_TO / np.mean(np.abs(act))
      act = act * mult

    print(f'Layer: {layer}; mean abs activations: {np.mean(np.abs(act))}')

    if MODE == 'train':
      hp = train_hp(act, pca_model)
      print('HPs: ', hp)
      model = make_model_from_hp(hp)

      error_train, batch_error_train = sparse_coding_model.train(
        model, act, hp['num_iter'], hp['nt_max'], hp['batch_size'], hp['eps'],
        verbose=hp['verbose'])

      if SAVE:    
        save_checkpoint(dst_p, model, hp, error_train, batch_error_train, pca_model=pca_model)
    
    else:
      model, hp = load_checkpoint(src_p)

      error_valid, batch_error_valid, valid_pred = sparse_coding_model.evaluate(
        model, act, hp['batch_size'], hp['nt_max'], hp['eps'])
      
      if SAVE:
        save_checkpoint(dst_p, model, hp, error_valid, batch_error_valid, valid_pred)