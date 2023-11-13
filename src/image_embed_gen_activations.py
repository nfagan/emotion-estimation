import utility
import torch
import torchvision
from torchvision.models.resnet import ResNet50_Weights
import numpy as np
import h5py
import os
import glob
from PIL import Image

"""
https://kozodoi.me/blog/20210527/extracting-features
"""

FEATURES = {}

def get_features(name):
  def hook(model, input, output):
    FEATURES[name] = output.detach()
  return hook

def get_device():
  return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
  dr = utility.get_dataroot()

  if True:
    # SPLIT = 'train'
    # ext = '.png'
    # im_dir = os.path.join(dr, 'flickr-faces/subset-emotion-balanced/train')
    # dst_p = os.path.join(dr, 'activations', 'resnet_image_embedding', 'train_expression_balanced')

    SPLIT = 'valid'
    ext = '.jpg'
    im_dir = os.path.join(dr, 'goldenberg_faces/images')
    dst_p = os.path.join(dr, 'activations', 'resnet_image_embedding', 'valid_expression_balanced')

  else:
    SPLIT = 'valid'
    # subdir = 'goldenberg_faces/reconstructed_images'
    subdir = 'goldenberg_faces/attended_images'
    dst_p = os.path.join(dr, 'activations', 'resnet_image_embedding', subdir)

    im_dir = os.path.join(dr, subdir)
    ext = '.png'

  src_im_ps = glob.glob(os.path.join(im_dir, f'*{ext}'))
  idents = [os.path.split(x)[1].replace(ext, '') for x in src_im_ps]
  device = get_device()
  weights = ResNet50_Weights.IMAGENET1K_V1
  tforms = weights.transforms()
  model = torchvision.models.resnet50(weights=weights)
  model.to(device)

  model.layer1.register_forward_hook(get_features('layer1'))
  # model.layer2.register_forward_hook(get_features('layer2'))
  # model.layer3.register_forward_hook(get_features('layer3'))
  # model.layer4.register_forward_hook(get_features('layer4'))

  is_chunked = False
  chunk_s = 512
  num_chunks = max(1, int(np.ceil(len(src_im_ps) / chunk_s)))
  if not is_chunked: num_chunks = 1

  for ci in range(num_chunks):
    i0 = ci * chunk_s
    i1 = min(i0 + chunk_s, len(src_im_ps))
    if not is_chunked: i1 = len(src_im_ps)
    im_ps = src_im_ps[i0:i1]
    ident_subset = idents[i0:i1]

    tot_acts = {}
    for i in range(len(im_ps)):
      print(f'{i+1} of {len(im_ps)}')

      im_p = im_ps[i]
      with Image.open(im_p) as im_f:
        im = torch.tensor(np.array(im_f))
      im = im.permute(2, 0, 1)[None, :, :, :]
      im = im.to(device)
      _ = model(tforms(im))
      
      for layer_name in FEATURES.keys():
        feat = FEATURES[layer_name].detach().cpu().numpy().flatten()
        if i == 0: tot_acts[layer_name] = np.zeros((len(im_ps), feat.size), dtype=np.float32)
        acts = tot_acts[layer_name]
        acts[i, :] = feat

    if True:
      if not is_chunked:
        chunk_dst_p = dst_p
      else:
        chunk_dst_p = os.path.join(dst_p, f'chunk_{ci}')
      os.makedirs(chunk_dst_p, exist_ok=True)
      for layer_name in tot_acts.keys():
        with h5py.File(os.path.join(chunk_dst_p, f'{layer_name}.h5'), 'w') as f:
          utility.h5_write_activations(f, tot_acts[layer_name], SPLIT, layer_name, ident_subset)