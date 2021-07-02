# written by Jaekwang Cha
# version 0.1

# ================== IMPORT CUSTOM LEARNING LIBRARIES ===================== #
from customs.train import train, test
from customs.dataset import load_dataset
from customs.model import load_model

# ================== TRAINING SETTINGS ================== #
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
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
parser.add_argument('--weight_histogram',action='store_true',            default=False, help='Enable histogram visualization')

# data setting
parser.add_argument('--val_rate',        default=0.2,                    type=float,    help='split rate for the validation data')
parser.add_argument('--transform',       default='default',              type=str,      help='choose the data transform type')

# training parameter setting
parser.add_argument('--n_epoch',         default=10,                     type=int,      help='number of total training iteration')
parser.add_argument('--batch_size',      default=32,                     type=int,      help='size of minibatch')
parser.add_argument('--test_batch_size', default=100,                    type=int,      help='size of test-minibatch')

# optimizer & scheduler setting
parser.add_argument('--lr',              default=0.03,                   type=float,    help='training learning rate')
parser.add_argument('--adapt_lr',        action='store_true',            default=False, help='activating adaptive learning rate')
parser.add_argument('--optimizer',       default='adam',                 type=str,      help='optimizer select')
parser.add_argument('--scheduler',       default='steplr',               type=str,      help='scheduler select')

# krotov learning setting
parser.add_argument('--k',               default=2,                      type=int,      help='krotov parameter')
parser.add_argument('--p',               default=2,                      type=int,      help='krotov parameter')
parser.add_argument('--n',               default=4.5,                    type=float,    help='krotov parameter')
parser.add_argument('--delta',           default=-0.4,                   type=float,    help='krotov parameter')

parser.add_argument('--BPA',             default=25,                     type=int,      help='BP activation')
opt = parser.parse_args()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def do_pca(n_components, data):
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    mat_data = np.asmatrix(pca.components_)
    print(mat_data.shape)
    return mat_data


# ===================== IMPORT PYTORCH LIBRARIES ================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(opt.seed)

# ================== GPU SETTINGS ================== #
def gpu_setup(opt):
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"

    if opt.multi_gpu != 0:
        print()
        print('Activating multi-gpu training mode')
        print(opt.multi_gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.multi_gpu)
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print()
        print('Activating single-gpu training mode')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using gpu number ' + str(opt.gpu))
    return use_cuda

# ======================= MAIN SCRIPT ============================= #
def main(opt):
    use_cuda = gpu_setup(opt)
    dataset_train, dataset_validation = load_dataset(opt, train=True)
    print('training data size: {}'.format(len(dataset_train)))
    print('validation data size: {}'.format(len(dataset_validation)))

    dataset_test = load_dataset(opt, train=False)
    print('test data size: {}'.format(len(dataset_test)))
    print()
    
    kwargs = {'num_workers': opt.num_worker, 'pin_memory': opt.pin_memory} if use_cuda else {}
    train_dataloader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
    validation_dataloader = DataLoader(dataset_validation, batch_size=opt.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(dataset_test, batch_size=opt.test_batch_size, shuffle=True, **kwargs)

    model = load_model(opt)
    if opt.multi_gpu != 0:
        model = torch.nn.DataParallel(model)
    model.to(opt.device)

    train(opt, model, train_dataloader, validation_dataloader)
    test(opt, model, test_dataloader)

if __name__ == '__main__':
    main(opt)
