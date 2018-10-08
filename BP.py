# -*- coding: utf-8 -*-
# https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

import numpy as np
from torch import nn
import torch as pt

inmat = np.array([0.1, 0.2, 0.7])
w_ih1 = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.7], [0.4, 0.3, 0.9]])
bias1 = np.array([1.0, 1.0, 1.0])
w_h1h2 = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.7], [0.6, 0.4, 0.8]])
w_h2o = np.array([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]])
target = np.array([1.0, 0.0, 0.0])
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=0)
# cross_ent = nn.BCELoss(reduction='sum')


def cross_ent_my(out, target):

  def ent(x, y):
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))

  return np.array([ent(x, y) for x, y in zip(out, target)])


def relu(x):
  return np.array([max(0, i) for i in x])


in_h1 = inmat @ w_ih1 + bias1
out_h1 = relu(in_h1)
in_h2 = out_h1 @ w_h1h2 + bias1
out_h2 = sigmoid(pt.tensor(in_h2)).numpy()
in_o = out_h2 @ w_h2o + bias1
out_o = softmax(pt.tensor(in_o)).numpy()
# out_o = np.array([0.2698 , 0.32235, 0.40784])
# Ept = cross_ent(pt.tensor(out_o), pt.tensor(target))
E_arr = cross_ent_my(out_o, target)
E_total = np.sum(E_arr)


def cross_ent_dev(x, y):
  return -1 * (y / x - (1 - y) / (1 - x))


for y, x in zip(target, out_o):
  print(cross_ent_dev(x, y))
