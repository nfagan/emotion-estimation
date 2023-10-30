import numpy as np
import h5py
import os
import glob
import utility

SRC_P = 'D:\\data\\changlab\\ilker_collab\\gpfs\\milgram\\pi\\chun\\ql225\\Memorability_and_CNN\\pca_activations'
DST_P = 'D:\\data\\changlab\\ilker_collab\\activations\\image_net'

def load_activations(ps):
  files = []
  for p in ps:
    files.extend(glob.glob(os.path.join(p, '*.npy')))
  
  act_list = [np.load(f) for f in files]
  file_names = [os.path.split(f)[1] for f in files]
    
  # see compute_pairwise_spc_distance_pca
  act_flat = np.asarray(act_list).squeeze(axis=1)
  act_flat = act_flat / np.quantile(np.abs(act_flat), 0.95)

  return act_flat, file_names

if __name__ == '__main__':
  layer_sets = [f'block{x+1}_pool' for x in range(5)] + ['fc1', 'fc2']
  # layer_sets = layer_sets[4:]

  # train, valid
  splits = ['Fillers', 'Targets']
  types = ['train', 'valid']

  for i in range(len(layer_sets)):
    print(f'{i+1} of {len(layer_sets)}')
    ls = layer_sets[i]

    tot_acts = []
    tot_file_names = []
    tot_types = []
    for j in range(len(splits)):
      acts, file_names = load_activations([os.path.join(SRC_P, splits[j], ls)])
      tot_acts.append(acts)
      tot_file_names.extend(file_names)
      tot_types.extend([types[j]] * len(file_names))

    tot_layers = [ls] * len(tot_file_names)
    tot_acts = np.concatenate(tot_acts)
    ascii_names = [x.encode('ascii', 'ignore') for x in tot_file_names]

    os.makedirs(DST_P, exist_ok=True)
    with h5py.File(os.path.join(DST_P, f'{ls}.h5'), 'w') as f:
      utility.h5_write_activations(f, tot_acts, tot_types, tot_layers, ascii_names)
