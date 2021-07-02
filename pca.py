from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from customs.dataset import load_dataset

parser = argparse.ArgumentParser()
# data setting
parser.add_argument('--train_method',    default='supervised',           type=str,      help='type of training: supervised(default), unsupervised, reinforce')
parser.add_argument('--opt_method',      default='backprop',             type=str,      help='type of optimization: backprop(default), local, mixed')
parser.add_argument('--learning_rule',   default='hebbian',              type=str,      help='hebbian(default), oja, gha, krotov')
parser.add_argument('--task',            default='classification',       type=str,      help='task of training: classification(default), regression')
parser.add_argument('--dataset',         default='mnist',                type=str,      help='dataset to use')
parser.add_argument('--model',           default='CNN',                  type=str,      help='model to use: CNN(default), Single_hebb, HebbCNN')
parser.add_argument('--seed',            default=42,                     type=int,      help='random seed (default: 42)')
parser.add_argument('--num_worker',      default=1,                      type=int,      help='number of dataloader worker')
parser.add_argument('--no_cuda',         action='store_true',            default=False, help='disables CUDA training')
parser.add_argument('--gpu',             default=0,                      type=str,      help='GPU-id for GPU to use')
parser.add_argument('--multi_gpu',       default=0,                      type=str,      help='GPU-ids for multi-GPU usage')
parser.add_argument('--pin_memory',      default=True,                   type=bool,     help='pin memory option selector')
parser.add_argument('--save_model',      action='store_true',            default=False, help='For Saving the current Model')
parser.add_argument('--save_path',       default=os.getcwd()+'/weights', type=str,      help='Where to save weights')
parser.add_argument('--log_path',        default=os.getcwd()+'/Logs',    type=str,      help='Where to save Logs')
parser.add_argument('--fast',            action='store_true',            default=False, help='Enable fast training mode')
parser.add_argument('--draw_pic',        action='store_true',            default=False, help='Enable visualization')
parser.add_argument('--get_gif',         action='store_true',            default=False, help='Enable visualization')


# data setting
parser.add_argument('--val_rate',        default=0.2,                    type=float,    help='split rate for the validation data')
parser.add_argument('--transform',       default='default',              type=str,      help='choose the data transform type')

# training parameter setting
parser.add_argument('--n_epoch',         default=10,                     type=int,      help='number of total training iteration')
parser.add_argument('--batch_size',      default=32,                     type=int,      help='size of minibatch')
parser.add_argument('--test_batch_size', default=32,                     type=int,      help='size of test-minibatch')

# optimizer & scheduler setting
parser.add_argument('--lr',              default=0.03,                   type=float,    help='training learning rate')
parser.add_argument('--adapt_lr',        action='store_true',            default=False, help='activating adaptive learning rate')
parser.add_argument('--optimizer',       default='adam',                 type=str,      help='optimizer select')
parser.add_argument('--scheduler',       default='steplr',               type=str,      help='scheduler select')

# krotov learning setting
parser.add_argument('--k',               default=2,                      type=int,      help='krotov parameter')
parser.add_argument('--p',               default=2,                      type=int,      help='krotov parameter')
parser.add_argument('--n',               default=4.5,                    type=float,      help='krotov parameter')
parser.add_argument('--delta',           default=-0.4,                   type=float,      help='krotov parameter')
opt = parser.parse_args()

import torch.nn as nn
import torch

from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def do_pca(n_components, data):
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def plot_component(pca):
    mat_data = np.asmatrix(pca.components_[0]).reshape(5,5)  #reshape images
    for i in range(len(pca.components_)-1):
        mat_data = np.concatenate((mat_data, np.asmatrix(pca.components_[i+1]).reshape(5,5)), axis=1)
    plt.imshow(mat_data, cmap='bwr'); #plot the data
    plt.xticks([]) #removes numbered labels on x-axis
    plt.yticks([]) #removes numbered labels on y-axis
    plt.show()   

if __name__ == '__main__':
    dataset_train, dataset_validation = load_dataset(opt, train=True)
    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    train_dataloader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True, **kwargs)

    inputs, classes = next(iter(train_dataloader))
    im2col = nn.Unfold((5,5), 1, 0, 1)
    X = im2col(inputs)
    X = X.permute(1,0,2).flatten(1).t().numpy()
    X1 = X[:,0:25]
    X2 = X[:,25:50]
    X3 = X[:,50:75]
    

    pca, X_pca = do_pca(25, X1)
    plot_component(pca)

    pca, X_pca = do_pca(25, X2)
    plot_component(pca)

    pca, X_pca = do_pca(25, X3)
    plot_component(pca)