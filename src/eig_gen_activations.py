import argparse
import os
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from PIL import Image
import h5py

from models.eig.networks.network import EIG
from models.eig.networks.network_classifier import EIG_classifier
from models.id.networks.network import VGG

from utils import config

CONFIG = config.Config()
IMAGE_FOLDER = 'C:\\Users\\nick\\source\\changlab\\ilker_collab\\EIG-faces\\infer_render_using_eig\\demo_images'
# OUTPUT_FOLDER = 'C:\\Users\\nick\\source\\changlab\\ilker_collab\\explore\\explore_sparse_coding\\output'
OUTPUT_FOLDER = None

DATASET_SPLIT = 'valid'
DST_P = os.path.join('D:\\data\\changlab\\ilker_collab\\activations\\eig', DATASET_SPLIT)

def load_image(image, size):
    image = image.resize(size)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = np.moveaxis(image, 2, 0)
    image = image.astype(np.float32)
    return image

models_d = {

    'eig' : EIG(),

}

image_sizes = {

    'eig' : (227, 227),

}


def main():
    parser = argparse.ArgumentParser(description='Predictions of the models on the neural image test sets')
    parser.add_argument('--imagefolder',  type=str, default='./demo_images/',
                        help='Folder containing the input images.')
    parser.add_argument('--segment', help='whether to initially perform segmentation on the input images.',
                       action='store_true')
    parser.add_argument('--addoffset', help='whether to add offset away from the image boundary to the output of the segmentation step.',
                       action='store_true')
    parser.add_argument('--resume', type = str, default='', 
                        help='Where is the model weights stored if other than where the configuration file specifies.')

    global args
    args = parser.parse_args()

    print("=> Construct the model...")
    model = models_d['eig']
    model.cuda()

    resume_path = args.resume
    if resume_path == '':
        resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'eig', 'checkpoint_bfm.pth.tar')
    checkpoint = torch.load(resume_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume_path, checkpoint['epoch']))

    # test
    outfile = None
    if OUTPUT_FOLDER is not None:
      os.makedirs(OUTPUT_FOLDER, exist_ok=True)
      outfile = os.path.join(OUTPUT_FOLDER, 'infer_output.hdf5')
    test(model, outfile)

def test(model, outfile):

    dtype = torch.FloatTensor

    path = IMAGE_FOLDER

    filenames = sorted(glob.glob(os.path.join(path, '*.png')))
    N = len(filenames)

    f3s = []
    f4s = []
    f5s = []

    latents = []
    attended = []
    for i in range(N):
        fname = filenames[i]
        print(f'{fname} | {i+1} of {N}')

        v = Image.open(fname)
        image = load_image(v, image_sizes['eig'])
        image = torch.from_numpy(image).type(dtype).cuda()
        image = image.unsqueeze(0)

        att, f3, f4, f5 = model(image, segment=args.segment, add_offset=args.addoffset and args.segment, test=True)

        f3s.append(f3.detach()[0].cpu().numpy().flatten())
        f4s.append(f4.detach()[0].cpu().numpy().flatten())
        f5s.append(f5.detach()[0].cpu().numpy().flatten())
        latents.append(f5.detach()[0].cpu().numpy().flatten())
        attended.append(att.detach()[0].cpu().numpy().flatten())

    asciiList = [n.split('/')[-1][:-4].encode("ascii", "ignore") for n in filenames]

    if outfile is not None:
      f = h5py.File(outfile, 'w')
      f.create_dataset('number_of_layers', data=np.array([2]))
      f.create_dataset('latents', data=np.array(latents))
      f.create_dataset('Att', data=np.array(attended))
      f.create_dataset('filenames', (len(asciiList), 1), 'S10', data=asciiList)
      f.close()

    layer_sets = [f3s, f4s, f5s]
    layer_names = ['f3', 'f4', 'f5']
    for i in range(len(layer_sets)):
      os.makedirs(DST_P, exist_ok=True)
      ln = layer_names[i]

      with h5py.File(os.path.join(DST_P, f'{ln}.h5'), 'w') as f:
        nf = len(asciiList)
        tot_types = [DATASET_SPLIT] * nf
        tot_layers = [layer_names[i]] * nf
        ascii_names = asciiList

        f.create_dataset('activations', data=np.array(layer_sets[i]))
        f.create_dataset('splits', (len(tot_types), 1), 'S10', data=tot_types)
        f.create_dataset('layers', (len(tot_layers), 1), 'S10', data=tot_layers)
        f.create_dataset('identifiers', (len(ascii_names), 1), 'S10', data=ascii_names)

if __name__ == '__main__':
    main()