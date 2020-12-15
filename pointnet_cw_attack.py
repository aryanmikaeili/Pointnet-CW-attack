import torch

import torch.nn as nn

import torch.optim as optim


import copy
from torch.autograd import Variable

from modelnet40_data import classes, PointnetDataset, load_data, Normalize, toTensor
from Pointnet_model import Pointnet
from torchvision import transforms

from torch.utils.data import DataLoader
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots



class L2_CW(object):
    def __init__(self, c=0.5, k=0, max_iter=100, lr=1e-2):
      self.c = c
      self.k = k
      self.lr = lr
      self.max_iter = max_iter

    def calc_f(self, logits, target_index):
      batch_size = logits.shape[0]
      class_size = logits.shape[1]
      logits_t = torch.masked_select(logits, torch.eye(class_size)[target_index].bool())
      logits_index_sorted = logits.argsort(descending=True, dim=1)
      max_logit_index = torch.zeros(batch_size)
      for i in range(batch_size):
        max_logit_index[i] = logits_index_sorted[i][0 if logits_index_sorted[i][0] != target_index[i] else 1]
      max_logit = torch.masked_select(logits, torch.eye(class_size)[max_logit_index.long()].bool())

      f = torch.clamp(max_logit - logits_t, min=self.k)

      return f

    def init_attack(self, points, noise_var = 0.01):


      init_pert = torch.randn(points.shape) * noise_var

      return init_pert

    def loss_function(self, adv_points, points, out, targets):
      f = self.calc_f(out, targets).sum()
      l2_loss = nn.MSELoss(reduction='sum')(adv_points, points)

      loss = l2_loss + self.c * f
      return loss, l2_loss, f

    def attack(self, model, points, targets):

      original_points = copy.deepcopy(points)

      w = Variable(self.init_attack(original_points), requires_grad=True)
      adv_points = w + original_points

      optimizer = optim.Adam([w], lr=self.lr)
      for i in range(self.max_iter):
        optimizer.zero_grad()
        model.zero_grad()
        out, _, _ = model(adv_points)
        loss, l2, f = self.loss_function(adv_points, points, out, targets)
        loss.backward()
        optimizer.step()

        adv_points = w + original_points


        if i % 10 == 0:
          print('loss: ', loss.item(), 'mse: ', l2.item(), 'f: ', f.item())

      out, _, _ = model(adv_points)
      labels = torch.argmax(out, dim=1)
      return adv_points, labels
    def attack2(self, model, points, targets):

      original_points = copy.deepcopy(points)
      adv_points = original_points + self.init_attack(original_points)
      adv_points = Variable(adv_points, requires_grad=True)

      optimizer = optim.Adam([adv_points], lr=self.lr)
      for i in range(self.max_iter):
        optimizer.zero_grad()
        model.zero_grad()
        out, _, _ = model(adv_points)
        loss, l2, f = self.loss_function(adv_points, points, out, targets)
        loss.backward()
        optimizer.step()


        if i % 10 == 0:
          print('loss: ', loss.item(), 'mse: ', l2.item(), 'f: ', f.item())

      out, _, _ = model(adv_points)
      labels = torch.argmax(out, dim=1)
      return adv_points, labels


def plot_original_adversarial(adv_pointcloud, original_pointcloud, true_label, adv_label):
  x_adv, y_adv, z_adv = adv_pointcloud.squeeze(0).detach().cpu().numpy()
  x_ori, y_ori, z_ori = original_pointcloud.squeeze(0).detach().cpu().numpy()

  trace_adv = go.Scatter3d(x=x_adv, y=z_adv, z=y_adv, mode='markers', marker=dict(color='red', opacity=0.5, size=4))
  trace_ori = go.Scatter3d(x=x_ori, y=z_ori, z=y_ori, mode='markers', marker=dict(color='blue', opacity=0.5, size=4))

  fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], subplot_titles=(list(classes.keys())[adv_label], list(classes.keys())[true_label]))
  fig.add_trace(trace_adv, row=1, col=1)
  fig.add_trace(trace_ori, row=1, col=2)
  fig.show()

if __name__ == '__main__':
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  Model_DIR = os.path.join(BASE_DIR, 'models')
  train_X, train_y = load_data()
  test_X, test_y = load_data(mode='test')

  attack_transform = transforms.Compose(
    [
      Normalize(),
      toTensor()]
  )


  testset = PointnetDataset(test_X, test_y, transform=attack_transform)
  testloader = DataLoader(testset, batch_size=32)
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
  else:
    device = torch.device('cpu')
    print('running on CPU')

  model = Pointnet(device = device)
  model.eval()

  model.load_state_dict(torch.load(os.path.join(Model_DIR, 'model.pt'), map_location=torch.device(device))['model_state_dict'])

  attack = L2_CW()

  sample = testset.__getitem__(1)
  sample_pointcloud, sample_label = sample[0].unsqueeze(0).transpose(1, 2).to(device), sample[1]

  adv_pointcloud, adv_label = attack.attack2(model, sample_pointcloud, torch.tensor([4]))

  plot_original_adversarial(adv_pointcloud, sample_pointcloud, sample_label, adv_label)
  a = 0







