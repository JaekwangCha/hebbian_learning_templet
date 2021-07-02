import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

class Conv2d_obs(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1,bias = False):
        super(Conv2d_obs, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    #def weight_histogram(self):

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return output

class Conv_hebb(nn.Module):
    # This is not actualy used now
    def init_params(self):
        self.custom_param = 1

class Conv2d_hebb(nn.Conv2d, Conv_hebb):
    '''
        ***Conv2d_hebb requires opt (argparse param. for training environment setup)
    '''
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1,bias = False):
        super(Conv2d_hebb, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.opt = opt
        torch.nn.init.normal_(self.weight)
        self.weight.requires_grad = False

        if bias:
            self.bias.requires_grad = False

        self.im2col = nn.Unfold(kernel_size, dilation, padding, stride)

    def krotov_activation(self, opt, input):
        _, indices = torch.topk(input, k=opt.k, dim=1)
        activations = torch.zeros(input.shape).to(opt.device)
        ind = torch.arange(input.shape[0]).unsqueeze(1).repeat(1, input.shape[2])
        activations[ind, indices[:,0], torch.arange(input.shape[2])] = 1.0
        for i in range(opt.k-1):
            activations[ind, indices[:,i+1], torch.arange(input.shape[2])] = opt.delta
        return activations

    def local_update(self, opt, pre_synaptic_activation, post_synaptic_activation):
        pre_synaptic_activation_mat = self.im2col(pre_synaptic_activation).transpose(1,2)
        post_synaptic_activation_mat = torch.flatten(post_synaptic_activation, start_dim=2)
        current_batch_size = pre_synaptic_activation_mat.shape[0]
        im2col_length = pre_synaptic_activation_mat.shape[1]

        if opt.learning_rule == 'hebbian':
            self.weight += opt.lr * torch.matmul(post_synaptic_activation_mat, pre_synaptic_activation_mat).mean(0).view(self.weight.shape)
            
        elif opt.learning_rule == 'oja':
            weight_mat = self.weight.flatten(1).repeat(current_batch_size,im2col_length,1,1).permute(0,2,1,3)
            yx = torch.matmul(post_synaptic_activation_mat, pre_synaptic_activation_mat)
            yyw = torch.matmul(post_synaptic_activation_mat.square().unsqueeze(2), weight_mat).squeeze()
            self.weight += opt.lr * (yx-yyw).mean(0).view(self.weight.shape)
            
        elif opt.learning_rule == 'gha':
            weight_mat = self.weight.flatten(1)#.repeat(current_batch_size,1,1)
            yx = torch.matmul(post_synaptic_activation_mat, pre_synaptic_activation_mat)
            yyw = torch.matmul(torch.matmul(post_synaptic_activation_mat, post_synaptic_activation_mat.transpose(1,2)).tril(), weight_mat)
            ds = (yx-yyw).mean(0).view(self.weight.shape)
            nc=torch.amax(torch.absolute(ds))
            prec=1e-30
            if nc<prec:
                nc=prec
            ds = torch.true_divide(ds, nc)
            self.weight.data = self.weight.data + opt.lr * ds

        elif opt.learning_rule == 'krotov':
            """Krotov-Hopfield Hebbian learning rule fast implementation.

            Original source: 
                https://github.com/DimaKrotov/Biological_Learning
                https://github.com/Joxis/pytorch-hebbian

            """
            pre_synaptic_activation_mat = pre_synaptic_activation_mat.transpose(1,2)
            weight_mat = self.weight.flatten(1)
            tot_input = torch.matmul(torch.sign(weight_mat) * torch.abs(weight_mat) ** (opt.p - 1), pre_synaptic_activation_mat)
            activations = self.krotov_activation(opt, tot_input)
            yx = torch.matmul(activations.permute(2,1,0), pre_synaptic_activation_mat.permute(2,0,1)).mean(0)
            yy = torch.sum(torch.mul(activations, tot_input), 0).mean(1).unsqueeze(1).repeat(1,weight_mat.shape[1])
            yyw = torch.mul(yy, weight_mat)
            ds = (yx-yyw).view(self.weight.shape)
            nc=torch.amax(torch.absolute(ds))
            prec=1e-30
            if nc<prec:
                nc=prec
            ds = torch.true_divide(ds, nc)
            self.weight += opt.lr * ds

        else:
            raise Exception('Needs proper setup param.')

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation)

        if self.opt.epoch < self.opt.BPA:
            self.local_update(self.opt, input, output)
        return output

