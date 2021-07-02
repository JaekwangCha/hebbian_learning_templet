from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .utils import weight2img
import matplotlib.pyplot as plt

def validation(opt, model, validation_dataloader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    data_iterator_validation = tqdm(validation_dataloader, desc='Epoch {} Validation...'.format(epoch))
    if opt.train_method == 'supervised' and opt.task == 'classification':
        with torch.no_grad():
            for i, (input, class_labels) in enumerate(data_iterator_validation):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                val_loss += F.nll_loss(output, class_labels, reduction='sum').item()
                correct += (predictions == class_labels).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            accuracy = correct / len(validation_dataloader.dataset) * 100
            print('Epoch {} Validation Results: val_loss {:.2f}, accuracy {:.2f}%'.format(epoch, val_loss, accuracy))
    
    elif opt.train_method == 'unsupervised' and opt.opt_method == 'local' and opt.task == 'classification':
        with torch.no_grad():
            for i, (input, class_labels) in enumerate(data_iterator_validation):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                val_loss += F.nll_loss(output, class_labels, reduction='sum').item()
                correct += (predictions == class_labels).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            accuracy = correct / len(validation_dataloader.dataset) * 100
            print('Epoch {} Validation Results: val_loss {:.2f}, accuracy {:.2f}%'.format(epoch, val_loss, accuracy*100))

    else:
        raise Exception('No proper setting')

def train(opt, model, train_dataloader, validation_dataloader):
    opt.epoch = 0
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.06, nesterov=True)
    #criterion = nn.CrossEntropyLoss()
    #scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)
    model.train()

    if opt.train_method == 'supervised' and opt.opt_method == 'backprop' and opt.task == 'classification':
        lr0 = opt.lr
        if opt.get_gif:
            import imageio
            image_list1 = []
            image_list2 = []

        for epoch in range(opt.n_epoch):
            data_iterator_training = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
            
            for i, (input, class_labels) in enumerate(data_iterator_training):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                optimizer.zero_grad()
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                
                # here needs loss function selector
                loss = F.nll_loss(output, class_labels)
                loss.backward()
                optimizer.step()

                correct = (predictions == class_labels).sum().item()
                accuracy = correct / opt.batch_size
                data_iterator_training.set_postfix(loss=loss.item(), accuracy = 100. * accuracy)

                if opt.get_gif and i%10==0:
                    tmp_img1 = weight2img(opt, model.conv1.weight)
                    tmp_img2 = weight2img(opt, model.conv2.weight)
                    pause_length = 1
                    for j in range(pause_length):
                        image_list1.append(tmp_img1)
                        image_list2.append(tmp_img2)
            
            validation(opt, model, validation_dataloader, epoch)
            if opt.adapt_lr:
                opt.lr=lr0*(1-(epoch+1)/opt.n_epoch)

        if opt.get_gif:
            imageio.mimwrite('animated_picture_w1.gif', image_list1)
            imageio.mimwrite('animated_picture_w2.gif', image_list2)    

        if opt.draw_pic:
            weight2img(opt, model.conv1.weight)
            weight2img(opt, model.conv2.weight)
    
    elif opt.train_method == 'unsupervised' and opt.opt_method == 'local' and opt.task == 'classification':
        lr0 = opt.lr
        if opt.get_gif:
            import imageio
            image_list = []

        for epoch in range(opt.n_epoch):
            data_iterator_training = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
            
            for i, (input, class_labels) in enumerate(data_iterator_training):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                if opt.fast:
                    continue
                data_iterator_training.set_postfix(weight_norm=torch.norm(model.conv1.weight.squeeze().flatten(1), dim=1).mean().item())
                
                if opt.get_gif:
                    tmp_img = weight2img(opt, model.conv1.weight)
                    pause_length = 1
                    for i in range(pause_length):
                        image_list.append(tmp_img)

            if opt.draw_pic:
                weight2img(opt, model.conv1.weight)
            
            if opt.adapt_lr:
                opt.lr=lr0*(1-(epoch+1)/opt.n_epoch)
        if opt.get_gif:
            imageio.mimwrite('animated_picture.gif', image_list)

    elif opt.train_method == 'supervised' and opt.opt_method == 'mixed' and opt.task == 'classification':
        lr0 = opt.lr
        if opt.get_gif:
            import imageio
            image_list1 = []
            image_list2 = []

        for epoch in range(opt.n_epoch):
            data_iterator_training = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
            opt.epoch = epoch

            if opt.epoch == opt.BPA:
                print('BP activated')
                model.conv1.weight.requires_grad = True
            
            for i, (input, class_labels) in enumerate(data_iterator_training):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)

                optimizer.zero_grad()
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()

                # here needs loss function selector
                loss = F.nll_loss(output, class_labels)
                loss.backward()
                optimizer.step()

                correct = (predictions == class_labels).sum().item()
                accuracy = correct / opt.batch_size
                data_iterator_training.set_postfix(loss=loss.item(), accuracy = 100. * accuracy)

                if opt.fast:
                    continue
                data_iterator_training.set_postfix(weight_norm=torch.norm(model.conv1.weight.squeeze().flatten(1), dim=1).mean().item(), 
                    accuracy = 100. * accuracy)

        

                if opt.get_gif and i%10==0:
                    tmp_img1 = weight2img(opt, model.conv1.weight)
                    tmp_img2 = weight2img(opt, model.conv2.weight)
                    pause_length = 1
                    for j in range(pause_length):
                        image_list1.append(tmp_img1)
                        image_list2.append(tmp_img2)

            if opt.adapt_lr:
                opt.lr=lr0*(1-(epoch+1)/opt.n_epoch)
            
            validation(opt, model, validation_dataloader, epoch)

        if opt.get_gif:
            imageio.mimwrite('animated_picture_w1.gif', image_list1)
            imageio.mimwrite('animated_picture_w2.gif', image_list2)

        if opt.draw_pic:
            weight2img(opt, model.conv1.weight)
            weight2img(opt, model.conv2.weight)

            
            
    else:
        raise Exception('No proper setting')

def test(opt, model, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    data_iterator_test = tqdm(test_dataloader, desc='Testing...')
    if opt.train_method == 'supervised' and opt.task == 'classification':
        with torch.no_grad():
            for i, (input, class_labels) in enumerate(data_iterator_test):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                test_loss += F.nll_loss(output, class_labels, reduction='sum').item()
                correct += (predictions == class_labels).sum().item()
            test_loss = test_loss / len(test_dataloader.dataset)
            accuracy = correct / len(test_dataloader.dataset) * 100
            print('Test Results: loss {:.2f}, accuracy {:.2f}%'.format(test_loss, accuracy))

    elif opt.train_method == 'unsupervised' and opt.opt_method == 'local' and opt.task == 'classification':
        with torch.no_grad():
            for i, (input, class_labels) in enumerate(data_iterator_test):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                test_loss += F.nll_loss(output, class_labels, reduction='sum').item()
                correct += (predictions == class_labels).sum().item()
            test_loss = test_loss / len(test_dataloader.dataset)
            accuracy = correct / len(test_dataloader.dataset) * 100
            print('Test Results: loss {:.2f}, accuracy {:.2f}%'.format(test_loss, accuracy))

    data_iterator_test = tqdm(test_dataloader, desc='Make Histogram...')
    if opt.weight_histogram:
        with torch.no_grad():
            im2col = nn.Unfold((5,5), dilation=1, padding=0, stride=1)
            weight_histogram = np.zeros(model.conv1.weight.shape[0])
            soft_histogram = np.zeros(model.conv1.weight.shape[0])
            
            for i, (input, class_labels) in enumerate(data_iterator_test):
                input  = input.to(opt.device)
                input_mat = im2col(input).permute(1,0,2).flatten(1)
                input_mat = input_mat[0:25,:]
                weight_mat = model.conv1.weight.flatten(1)
                weight_mat = weight_mat[:,0:25]
                tot_input = torch.abs(torch.matmul(weight_mat, input_mat))
                sm_input = F.softmax(tot_input, dim=0).mean(dim=1)
                sm_input = sm_input.squeeze().detach().cpu().numpy()
                soft_histogram = (soft_histogram * i + sm_input) / (i+1)

                _, indices = torch.topk(tot_input, k=1, dim=0)

                indices = indices.squeeze().detach().cpu().numpy()
                indices = np.random.choice(indices, int(len(indices)/100))
                
                for j in range(len(indices)):
                    weight_histogram[indices[j]] += 1
            
        np.save('./weight_histogram_{}_{}'.format(opt.n_epoch, opt.model), weight_histogram)
        np.save('./soft_histogram_{}_{}'.format(opt.n_epoch, opt.model), soft_histogram)

