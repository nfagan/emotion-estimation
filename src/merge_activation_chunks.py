import utility
import h5py
import os
import numpy as np

SRC_P = 'D:\\data\\changlab\\ilker_collab\\activations\\arcface_recog\\train_expression_balanced'

if __name__ == '__main__':
  # layer_names = ['resnet_layer2', 'resnet_layer3', 'resnet_layer4', 'resnet_output', 'ReconNetWrapper_output']
  # layer_names = ['resnet_layer1']
  layer_names = ['layer1']
  # layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

  chunk_ps = next(os.walk(SRC_P))[1]
  chunk_ps = [x for x in chunk_ps if 'chunk_' in x]
  num_chunks = len(chunk_ps)
  chunks = [f'chunk_{x}' for x in range(num_chunks)]

  for layer in layer_names:
    dst_p = os.path.join(SRC_P, f'{layer}.h5')
    nr = 0
    act_dim = 0

    for i in range(len(chunks)):
      print(f'{i + 1} of {len(chunks)}')
      chunk = chunks[i]

      with h5py.File(os.path.join(SRC_P, chunk, f'{layer}.h5'), 'r') as fr:
        acts, _, _, _ = utility.load_activations(fr)
        if i == 0:
          act_dim = acts.shape[1]
        else:
          assert acts.shape[1] == act_dim
        nr += acts.shape[0]
    
    with h5py.File(dst_p, 'w') as fw:
      utility.h5_create_datasets(fw, (nr, act_dim))

      off = 0
      for i in range(len(chunks)):
        print(f'{i + 1} of {len(chunks)}')
        chunk = chunks[i]

        with h5py.File(os.path.join(SRC_P, chunk, f'{layer}.h5'), 'r') as fr:
          acts, splits, fnames, layers = utility.load_activations(fr)
          utility.h5_write_into_datasets(fw, acts, splits, layers, fnames, off)
          off += acts.shape[0]
