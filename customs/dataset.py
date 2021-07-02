from torchvision import datasets, transforms
import torch
import numpy as np

def transform(opt):
    if opt.transform == 'default':
        transform = transform=transforms.ToTensor()
    return transform

def mnist(opt, train=True):
    if train:
        dataset = datasets.MNIST(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)
        validation_size = int(opt.val_rate * len(dataset))
        train_size = len(dataset) - validation_size
        return torch.utils.data.random_split(dataset, [train_size, validation_size])
    else:
        return datasets.MNIST(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)

def get_dataset_stats(limit):
    """
    reference: https://github.com/GabrieleLagani/HebbianLearningThesis
    """
    MEAN_KEY = 'mean'
    STD_KEY = 'std'
    ZCA_KEY = 'zca'

    print("Computing statistics on dataset[0:" + str(limit) + "] (this might take a while)")
    cifar10 = datasets.CIFAR10(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=True, download=True)
    X = cifar10.data[0:limit] # X is M x N (M = limit: samples, N = 3072: variables per dataset sample)
    X = X / 255.
    # Compute mean and st. dev. and normalize the data to zero mean and unit variance
    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std = X.std(axis=(0, 1, 2), keepdims=True)
    X = (X - mean)/std
    # Transpose image tensors dimensions in order to put channel dimension in pos. 1, as expected by pytorch
    X = X.transpose(0, 3, 1, 2)
    # Reshape image tensors from shape 32x32x3 to vectors of size 32*32*3=3072
    X = X.reshape(limit, -1)
    # Compute ZCA matrix
    cov = np.cov(X, rowvar=False)
    U, S, V = np.linalg.svd(cov)
    SMOOTHING_CONST = 1e-1
    zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + SMOOTHING_CONST)), U.T))
    stats = {MEAN_KEY: mean.squeeze().tolist(), STD_KEY: std.squeeze().tolist(), ZCA_KEY: torch.from_numpy(zca).float()}
    return stats[MEAN_KEY], stats[STD_KEY], stats[ZCA_KEY]

def cifar10(opt, train=True):
    '''
    mean, std, zca = get_dataset_stats(50000)
    T = transforms.Compose([
			# The first transform is ToTensor, which transforms the raw CIFAR10 data to a tensor in the form
			# [depth, width, height]. Additionally, pixel values are mapped from the range [0, 255] to the range [0, 1]
			transforms.ToTensor(),
			# The Normalize transform subtracts mean values from each channel (passed in the first tuple) and divides each
			# channel by std dev values (passed in the second tuple). In this case we bring each channel to zero mean and
			# unitary std dev, i.e. from range [0, 1] to [-1, 1]
			transforms.Normalize(mean, std)
		])
    T = transforms.Compose([T, transforms.LinearTransformation(zca, torch.zeros(zca.size(1)))])
    '''

    if train:
        dataset = datasets.CIFAR10(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)
        #dataset = datasets.CIFAR10(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=T, download=True)
        validation_size = int(opt.val_rate * len(dataset))
        train_size = len(dataset) - validation_size
        return torch.utils.data.random_split(dataset, [train_size, validation_size])
    else:
        return datasets.CIFAR10(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)

def load_dataset(opt, train=True):
    if opt.dataset == 'mnist':
        return mnist(opt, train)
    elif opt.dataset == 'cifar10':
        return cifar10(opt, train)
    else:
        print('err: there is no dataset')