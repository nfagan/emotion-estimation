import utility
import h5py
import os
import numpy as np

SRC_P = 'D:\\data\\changlab\\ilker_collab\\activations\\d3dfr\\train_expression_balanced'

if __name__ == '__main__':
  num_chunks = 9
  layer_names = ['resnet_layer2', 'resnet_layer3', 'resnet_layer4', 'resnet_output', 'ReconNetWrapper_output']
  # layer_names = ['ReconNetWrapper_output']

  chunks = [f'chunk_{x}' for x in range(num_chunks)]
  
  for layer in layer_names:
    for i in range(len(chunks)):
      print(f'{i + 1} of {len(chunks)}')
      chunk = chunks[i]

      with h5py.File(os.path.join(SRC_P, chunk, f'{layer}.h5'), 'r') as f:
        acts, splits, fnames, layers = utility.load_activations(f)

      if i == 0:
        tot_acts, tot_splits, tot_names, tot_layers = acts, splits, fnames, layers
      else:
        tot_acts = np.concatenate((tot_acts, acts))
        tot_splits = np.concatenate((tot_splits, splits))
        tot_names = np.concatenate((tot_names, fnames))
        tot_layers = np.concatenate((tot_layers, layers))

    dst_p = os.path.join(SRC_P, f'{layer}.h5')
    with h5py.File(dst_p, 'w') as f:
      utility.h5_write_activations(f, tot_acts, tot_splits, tot_layers, tot_names)

