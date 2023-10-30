import sparse_coding_model
import utility
import numpy as np
import scipy.io
import os

data_root = utility.get_dataroot()

MODE = 'valid'
SELECT_SPLIT = 'valid'
SAVE = True

DATASET = 'd3dfr'
VARIANT = '_var_subsample_2048'
VARIANT = '_expression_balanced_var_subsample_1000'
# VARIANT = ''
LAYERS = [f'resnet_layer{x+1}' for x in range(4)] + ['resnet_output', 'ReconNetWrapper_output']
LAYERS = LAYERS[1:]
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

def save_checkpoint(save_p, model, hp, error, batch_error, prediction=None):
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

def train_hp(act, layer):
  """
  see compute_pairwise_spc_distance_pca
  """
  lr = 1e-2
  # lr = 1e-3 # @NOTE: this is lower than Qi's, which was 1e-2
  hp = {}
  hp['codeword_dim'] = act.shape[1]
  hp['num_codewords'] = hp['num_units'] = 500
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
    acts = f'{DATASET}/{MODE}{VARIANT}/{layer}'
    cp_dir = 'sc_checkpoints' if MODE == 'train' else 'sc_eval'
    dst_p = os.path.join(data_root, cp_dir, acts)

    act, split, _, _ = utility.load_activations(os.path.join(data_root, 'activations', f'{acts}.h5'))
    act = act[split == SELECT_SPLIT, :]

    if RESCALE_MEAN_ACTIVATIONS_TO is not None:
      mult = RESCALE_MEAN_ACTIVATIONS_TO / np.mean(np.abs(act))
      act = act * mult

    print(f'Layer: {layer}; mean abs activations: {np.mean(np.abs(act))}')

    if MODE == 'train':
      hp = train_hp(act, layer)
      model = make_model_from_hp(hp)

      error_train, batch_error_train = sparse_coding_model.train(
        model, act, hp['num_iter'], hp['nt_max'], hp['batch_size'], hp['eps'],
        verbose=hp['verbose'])

      if SAVE:    
        save_checkpoint(dst_p, model, hp, error_train, batch_error_train)
    
    else:
      src_p = os.path.join(data_root, 'sc_checkpoints', acts.replace('valid', 'train')) # load train checkpoint
      model, hp = load_checkpoint(src_p)

      error_valid, batch_error_valid, valid_pred = sparse_coding_model.evaluate(
        model, act, hp['batch_size'], hp['nt_max'], hp['eps'])
      
      if SAVE:
        save_checkpoint(dst_p, model, hp, error_valid, batch_error_valid, valid_pred)