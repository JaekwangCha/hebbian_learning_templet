import torch.nn as nn
import torch.nn.functional as F
from .layer import Conv2d_hebb, Conv2d_obs

class Ref_CNN(nn.Module):
    def __init__(self):
        super(Ref_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,96,5)
        self.conv2 = nn.Conv2d(96,128,3)
        self.conv3 = nn.Conv2d(128,192,3)
        self.conv4 = nn.Conv2d(192,256,3)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(192)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(300)
        
        self.fc1 = nn.Linear(3*3*256, 300)
        self.fc2 = nn.Linear(300, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = x.view(-1, 3*3*256)
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        x = nn.Dropout(p=0.5)(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        #return x

class Ref_HebbCNN(nn.Module):
    def __init__(self, opt):
        super(Ref_HebbCNN, self).__init__()
        self.conv1_hebb = Conv2d_hebb(opt,3,96,5,1)
        self.conv2 = nn.Conv2d(96,128,3,1)
        self.conv3 = nn.Conv2d(128,192,3,1)
        self.conv4 = nn.Conv2d(192,256,3,1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(192)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(300)
        
        self.fc1 = nn.Linear(3*3*256, 300)
        self.fc2 = nn.Linear(300, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1_hebb(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = x.view(-1, 3*3*256)
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        x = nn.Dropout(p=0.5)(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d_obs(3,40,5,1)
        self.conv2 = Conv2d_obs(40,60,5,1)
        self.fc1 = nn.Linear(5*5*60, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class Single_hebb_layer(nn.Module):
    def __init__(self, opt):
        super(Single_hebb_layer, self).__init__()
        self.opt = opt
        self.conv1_hebb = Conv2d_hebb(opt,3,1000,32,1)
        self.fc = nn.Linear(1000,10)
        
    def forward(self, x):
        x = self.conv1_hebb(x)
        x = x.squeeze()
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class HebbCNN(nn.Module):
    def __init__(self, opt):
        super(HebbCNN, self).__init__()
        self.opt = opt
        #self.conv1_hebb = nn.Conv2d(3,20,5,1)
        self.conv1 = Conv2d_hebb(opt,3,40,5,1)
        #self.conv2_hebb = Conv2d_hebb(opt,20,60,5,1)
        self.conv2 = nn.Conv2d(40,60,5,1)
        self.fc1 = nn.Linear(5*5*60, 500)
        #self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*50)
        x = x.view(-1, 5*5*60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_model(opt):
    if opt.model == 'CNN':
        return CNN()
    elif opt.model == 'Single_hebb':
        return Single_hebb_layer(opt)
    elif opt.model == 'HebbCNN':
        return HebbCNN(opt)
    else:
        raise Exception('model is not specified')
