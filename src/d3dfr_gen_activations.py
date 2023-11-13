"""This script is modified from the `test.py` test script from Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.networks import define_net_recog
from util.visualizer import MyVisualizer
from util.preprocess import align_img, estimate_norm_torch
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
import h5py
from utility import h5_write_activations, get_dataroot

SPLIT = 'train'
DST_P = os.path.join(get_dataroot(), 'activations', 'd3dfr', f'{SPLIT}_expression_balanced')
# DST_IM_P = os.path.join(get_dataroot(), 'd3dfr_face_recon', 'results', f'{SPLIT}_expression_balanced')

DST_IM_P = os.path.join(get_dataroot(), 'd3dfr_face_recon', 'results', 'images', 'epoch_20_000000')

SAVE_MODEL_OUTPUT = False
SAVE_PRED_MASK = False
SAVE_RECON_ACTIVATIONS = False
SAVE_RECOG_ACTIVATIONS = True
# SUBSAMPLE_SIZE = 1000
SUBSAMPLE_SIZE = None
CHUNK_SIZE = 256

FEATURES = {}

def get_features(name):
    def hook(model, input, output):
        FEATURES[name] = output.detach()
    return hook

def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    
    lm_path_e = [x for x in lm_path if os.path.isfile(x)]
    im_path_e = [im_path[i] for i in range(len(lm_path)) if os.path.isfile(lm_path[i])]

    return im_path_e, lm_path_e

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)
    visualizer.save_dir = DST_IM_P

    """
    recognet
    """
    recog_model = define_net_recog(
        'r50',
        pretrained_path=os.path.join(get_dataroot(), 'arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'),
        input_size=112
    )
    recog_model.to(device)
    recog_model.net.layer1.register_forward_hook(get_features('layer1'))
    # recog_model.net.layer2.register_forward_hook(get_features('layer2'))
    # recog_model.net.layer3.register_forward_hook(get_features('layer3'))
    # recog_model.net.layer4.register_forward_hook(get_features('layer4'))
    """
    end recognet
    """

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    layer_names = [
        'resnet_layer1', 'resnet_layer2',
        'resnet_layer3', 'resnet_layer4', 'resnet_output', 'ReconNetWrapper_output'
    ]
    #   @TODO: Re-add resnet_layer1
    # layer_names = layer_names[:1]
    layer_names = ['resnet_layer1']

    num_ims = len(im_path)
    # num_ims = min(10, num_ims)
    chunk_size = CHUNK_SIZE
    num_chunks = int(np.ceil(num_ims / chunk_size))

    for chunk in range(0, num_chunks):
        print(f'Chunk {chunk}')
        layer_outputs = {name: [] for name in layer_names}

        i0 = chunk * chunk_size
        i1 = min(i0 + chunk_size, num_ims)

        im_names = []
        tot_acts = {}
        for i in range(i0, i1):
            print('\t', i, im_path[i])
            img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
            im_names.append(img_name)
            if not os.path.isfile(lm_path[i]):
                print("%s is not found !!!"%lm_path[i])
                continue
            im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
            data = {
                'imgs': im_tensor,
                'lms': lm_tensor
            }
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference

            """
            recog net
            """
            trans_m = estimate_norm_torch(model.pred_lm, model.input_img.shape[-2])
            recog_out = recog_model(model.input_img, trans_m)
            """
            end recog net
            """

            visuals = model.get_current_visuals()  # get image results
            visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
                save_results=SAVE_MODEL_OUTPUT, count=i, name=img_name, add_image=False)
            
            if SAVE_PRED_MASK:
                pm = (model.pred_mask.detach().cpu().numpy().squeeze() * 255.).astype(np.uint8)
                savemat(os.path.join(DST_IM_P, f'{img_name}_pred_mask.mat'), {'pred_mask': pm})

            if SAVE_MODEL_OUTPUT:
                save_dir = visualizer.img_dir if visualizer.save_dir is None else visualizer.save_dir
                model.save_mesh(os.path.join(save_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
                model.save_coeff(os.path.join(save_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

            if SAVE_RECON_ACTIVATIONS:
                for name in layer_names:
                    v = model.acts[name].cpu().flatten(1).numpy()
                    if SUBSAMPLE_SIZE is not None:
                        v = v[:, :SUBSAMPLE_SIZE]
                    layer_outputs[name].append(v)

            """
            recog net
            """
            for layer_name in FEATURES.keys():
                feat = FEATURES[layer_name].detach().cpu().numpy().flatten()
                if i == i0: tot_acts[layer_name] = np.zeros((i1 - i0, feat.size), dtype=np.float32)
                acts = tot_acts[layer_name]
                acts[i - i0, :] = feat
            """
            end recog net
            """
        if SAVE_RECOG_ACTIVATIONS:
            chunk_p = os.path.join(DST_P.replace('d3dfr', 'arcface_recog'), f'chunk_{chunk}')
            os.makedirs(chunk_p, exist_ok=True)
            for layer_name in FEATURES.keys():
                with h5py.File(os.path.join(chunk_p, f'{layer_name}.h5'), 'w') as f:
                    h5_write_activations(f, tot_acts[layer_name], SPLIT, layer_name, im_names)

        if SAVE_RECON_ACTIVATIONS:
            chunk_p = os.path.join(DST_P, f'chunk_{chunk}')
            os.makedirs(chunk_p, exist_ok=True)
            acts = [np.concatenate(layer_outputs[x]) for x in layer_names]
            for i in range(len(acts)):
                ln = layer_names[i]
                with h5py.File(os.path.join(chunk_p, f'{ln}.h5'), 'w') as f:
                    h5_write_activations(f, acts[i], SPLIT, ln, im_names)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt,opt.img_folder)
    
