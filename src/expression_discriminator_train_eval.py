import utility
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import os
import sys

class ClassifierNet(nn.Module):
  def __init__(self, input_dim, hidden_dim = 512):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, 2)
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

  def forward(self, v0):
    v1 = nn.functional.relu(self.fc1(v0))
    v2 = self.fc2(v1)
    return v2
  
def get_device():
  return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def checkpoint_p(dr):
  return os.path.join(dr, 'ed_checkpoints', 'checkpoint.pth')

def eval_p(dr):
  return os.path.join(dr, 'ed_eval', 'checkpoint.mat')

def save_checkpoint(model, subsel, dr):
  cp = checkpoint_p(dr)
  os.makedirs(os.path.split(cp)[0], exist_ok=True)
  torch.save({
    'state_dict': model.state_dict(),
    'input_dim': model.input_dim,
    'hidden_dim': model.hidden_dim,
    'subselect_features': subsel,
  }, checkpoint_p(dr))

def do_feat_subselect(act):
   # only first 224 components of activations (alpha, beta, gamma, see deep 3d recon paper)
  return act[:, :224]

def load_model(dr):
  cp = torch.load(checkpoint_p(dr))
  model = ClassifierNet(cp['input_dim'], hidden_dim=cp['hidden_dim'])
  model.load_state_dict(cp['state_dict'])
  return model, cp

def eval_label():
  dr = utility.get_dataroot()
  device = get_device()
  model, cp = load_model(dr)
  model.to(device)
  model.eval()
  subsel_features = cp['subselect_features']

  act, split, ident, _ = utility.load_activations(
    os.path.join(dr, 'activations/d3dfr/train_complete', 'ReconNetWrapper_output.h5'))
  if subsel_features: act = do_feat_subselect(act)

  act = torch.tensor(act).to(device)
  
  res = model(act)
  pred_ps = predict_ps(res)
  prediction = torch.argmax(pred_ps, dim=1)

  prediction = prediction.detach().cpu().numpy()
  pred_ps = pred_ps.detach().cpu().numpy()

  save_p = eval_p(dr)
  os.makedirs(os.path.split(save_p)[0], exist_ok=True)

  scipy.io.savemat(save_p, {
    'prediction': prediction,
    'prediction_ps': pred_ps,
    'identifier': ident,
    'label': ['negative', 'positive']
  })

def predict_ps(res):
  return nn.functional.softmax(res, dim=1)

def train():
  subsel_features = False  # only first 224 components of activations (alpha, beta, gamma, see deep 3d recon paper)

  dr = utility.get_dataroot()
  device = get_device()

  im_info = utility.load_goldenberg_image_info(os.path.join(dr, 'goldenberg_faces/image_info_struct.mat'))
  valences = im_info['valence']
  idents = np.array(im_info['identifier'])
  valence_index = np.array(np.array(valences) == 'positive', dtype=int)

  act, split, ident, _ = utility.load_activations(
    os.path.join(dr, 'activations/d3dfr/valid_var_subsample', 'ReconNetWrapper_output.h5'))
  
  if subsel_features: act = do_feat_subselect(act)
  
  ok_ident = [x in idents for x in ident]
  act, split, ident = act[ok_ident, :], split[ok_ident], ident[ok_ident]

  # arrange valences by identifier in activation set
  ident_index = [np.argwhere(idents == x)[0, 0] for x in ident]
  valence_index = valence_index[ident_index]

  targets = torch.tensor(valence_index, dtype=torch.int64).to(device)

  """
  train / valid split
  """
  ti = []
  vi = []
  for i in range(2):
    val_mask = np.argwhere(valence_index == i)[:, 0]
    sub_sel = np.random.choice(len(val_mask), int(np.floor(len(val_mask) * 0.7)), replace=False)
    ti.append(val_mask[sub_sel])
    vi.append(np.setdiff1d(val_mask, val_mask[sub_sel]))
  ti, vi = np.concatenate(ti), np.concatenate(vi)

  """
  train
  """
  
  model = ClassifierNet(act.shape[1]).to(device)
  opt = optim.Adam(model.parameters(), lr=1e-4)

  batch_size = 64
  num_iter = 1000
  for i in range(num_iter):
    acti = ti[np.random.choice(ti.shape[0], (batch_size,), replace=False)]

    act_subset = torch.tensor(act[acti, :]).to(device)
    targ_subset = targets[acti]

    res = model(act_subset)
    pred = torch.argmax(nn.functional.softmax(res, dim=1), dim=1)
    acc = torch.sum(targ_subset == pred).type(torch.float64) / len(pred)

    loss = nn.functional.cross_entropy(res, targ_subset)
    print('Loss: {:.3f}; Acc: {:.3f}'.format(loss.item(), acc.item()))

    opt.zero_grad()
    loss.backward()
    opt.step()
  
  model.eval()
  res_eval = model(torch.tensor(act[vi, :]).to(device))
  loss = nn.functional.cross_entropy(res_eval, targets[vi])

  pred = torch.argmax(nn.functional.softmax(res_eval, dim=1), dim=1)
  acc = torch.sum(targets[vi] == pred).type(torch.float64) / len(pred)
  print('Valid loss: {:.3f}; Acc: {:.3f}'.format(loss.item(), acc.item()))

  if True:    
    save_checkpoint(model, subsel_features, dr)

if __name__ == '__main__':
  if sys.argv[1] == 'train':
    train()

  elif sys.argv[1] == 'eval':
    eval_label()

  else:
    assert False