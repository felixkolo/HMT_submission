# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:42:37 2020

@author: Felix
"""
#%% Initialization
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# from numpy import inf
# from skimage import io, transform
import pandas as pd
# import pathlib
# import torchvision
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# import scipy.stats
# import pylab
from copy import deepcopy
# import torchvision.models as models
from PIL import Image
# from PIL import ImageStat
# import warnings
# import time
# import sys
import math
# import pickle
import datetime
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    #
#
__all__ = ['ResNet', 'resnet18']
#
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}
#
comment = ['Start: {}'.format(datetime.datetime.now())]
# comment.append('This run uses native resnet18')
comment.append('This run uses an own Network architecture')
# comment.append('Layer 1: CONV   S=3, P=7 -> BN  -> RELU -> DROPOUT: 0.1 -> POOL')
# comment.append('Layer 2: CONV   S=1, P=5 -> BN  -> RELU -> DROPOUT: 0.2 -> POOL')
# comment.append('Layer 3: CONV   S=1, P=3 -> BN  -> RELU -> DROPOUT: 0.2 -> POOL')
# comment.append('Layer 4: CONV   S=1, P=1 -> BN  -> RELU -> DROPOUT: 0.2 -> POOL')
comment.append('Layer 1: CONV   S=3, P=7 -> BN  -> RELU -> POOL')
comment.append('Layer 2: CONV   S=1, P=5 -> BN  -> RELU -> POOL')
comment.append('Layer 3: CONV   S=1, P=3 -> BN  -> RELU -> POOL')
comment.append('Layer 4: CONV   S=1, P=1 -> BN  -> RELU -> POOL')
comment.append('Layer 5: FC                     -> RELU -> DROPOUT: 0.5')
# comment.append('Layer 4: FC                     -> RELU')
comment.append('Layer 6: FC                     ')
comment.append('This run uses num_workers = 2')
#
# set directories
simdir = os.path.join(os.getcwd(), '..','Sim')
# simdir = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit/new-geometry/Sim/"
train_dir = os.path.join(simdir, "Training")
test_dir = os.path.join(simdir, "Test")
imgpth = os.path.join(train_dir, 'image')
imgpthtest = os.path.join(test_dir, 'image')
imagepth='image'
#
# set some hyperparameters
num_epochs = 4  # number of times a full dataset is sampled
num_bins = 40   # number of bins in the dataset histogram
batch_size = {'Training': 4, 'Test': 4} # batch size for data loader
lr = 0.001  # learning rate of optimizer
#
comment.append('Batch size: {}, number of epochs: {}'.format(batch_size['Training'], num_epochs))
comment.append('The learning rate is {}.'.format(lr))
#
def getdirname(simdir):
    for i in range(1, 10000):
        if not os.path.exists(os.path.join(simdir, 'Evaluation', str(i))):
            break
    return(str(i))
    #
#
# set train and test boolean variables
# load_model_bool = True
load_model_bool = False
train_bool = True
# train_bool = False
save_model_bool = True
# save_model_bool = False
test_bool = True                # load_model must (if run exclusively) also be True
# test_bool = False
# printlog = False
printlog = True
# load_cancelled_model = True
load_cancelled_model = False
#
if load_model_bool == True:
    dirName = '26' # has to be defined when loading a model
else:
    dirName = getdirname(simdir)
    #
# set more paths
currentsim = os.path.join(simdir, 'Evaluation', dirName)
logpath = os.path.join(currentsim, '1-log.txt')
testlogpath = os.path.join(currentsim, '4-testing.txt')
model_name = '8-mycnn.pth'
bestmodel_name = '8-mycnn-best.pth'
model_path = os.path.join(currentsim, model_name)
checkpoint_path = os.path.join(currentsim, 'checkpoint.tar')
objects_path = os.path.join(currentsim, 'objects.tar')
#
# create directory
if train_bool==True and load_model_bool==False:
    if not os.path.exists(currentsim) and int(dirName) < 65535:
        os.makedirs(currentsim)
        print("Directory ", dirName,  " Created ")
    #        break
    else:
        print("Directory ", dirName,  " already exists")
elif train_bool==True and load_model_bool==True:
    comment.append('This run further trains the previous model')
    #
# check if cuda device available (run on gpu if available -> much faster)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
comment.append('This ran on {}'.format(device))
#
#%% Definition of functions

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def printf(statement, printlog, logpath):
    if printlog == True:
        with open(logpath, 'a') as f:
            print(statement, file=f)
            print(statement)
    else:
        print(statement)

def print_results(i, output, labels, criterion):
    for j, out in enumerate(output):
        dev_hor = 100* abs((out[0].item() - labels[j][0].item()) / labels[j][0].item())
        dev_ver = 100* abs((out[1].item() - labels[j][1].item()) / labels[j][1].item())
        printf('Batch: '+str(i)+'\tPredicted:\thorizontal:\t' + str(round(out[0].item(), 3))+ ',\tactual: ' + str(round(labels[j][0].item(), 3))+ ',\tDeviation horizontal:\t'+str(round(dev_hor, 1))+' %', printlog, logpath)
        printf('\t\tPredicted:\tvertical: \t' + str(round(out[1].item(), 3))+ ',\tactual: ' + str(round(labels[j][1].item(), 3))+ ',\tDeviation vertical:  \t'+ str(round(dev_ver, 1))+' %', printlog, logpath)
    return dev_hor, dev_ver

def MAPELoss(output, labels):
    return torch.mean(torch.abs((labels - output) / labels))

# save loss in text files
def savelosshistoryfiles(loss_phase, currentsim):
    lossfilename = '2-loss_training_history.txt'
    lossfilename_old = '2-loss_training_history_old.txt'
    pfad = os.path.join(currentsim, lossfilename)
    pfad_old = os.path.join(currentsim, lossfilename_old)
    if os.path.exists(pfad):
        if os.path.exists(pfad_old):
            os.remove(pfad_old)
            os.rename(pfad, pfad_old)
            f=open(pfad, 'w+')
            printf(("File ", lossfilename,  " renamed to ", lossfilename_old), printlog, logpath)
            printf(("File ", lossfilename,  " created. "), printlog, logpath)
        else:
            os.rename(pfad, pfad_old)
            f=open(pfad, 'w+')
            printf(("File ", lossfilename,  " renamed to ", lossfilename_old), printlog, logpath)
            printf(("File ", lossfilename,  " created. "), printlog, logpath)
    else:
        f=open(pfad, 'w')
        printf(("File ", lossfilename,  " created."), printlog, logpath)
    f.write('** Loss train history\n')
    for row in loss_phase['Training']:
        np.savetxt(f, np.array([str(float(row))]), fmt="%s")
        printf(np.array([str(float(row))]), printlog, logpath)
    f.close()
    lossfilename = '3-loss_test_history.txt'
    lossfilename_old = '3-loss_test_history_old.txt'
    pfad = os.path.join(currentsim, lossfilename)
    pfad_old = os.path.join(currentsim, lossfilename_old)
    if os.path.exists(pfad):
        if os.path.exists(pfad_old):
            os.remove(pfad_old)
            os.rename(pfad, pfad_old)
            f=open(pfad, 'w+')
            printf(("File ", lossfilename,  " renamed to "), lossfilename_old, printlog, logpath)
            printf(("File ", lossfilename,  " created. "), printlog, logpath)
        else:
            os.rename(pfad, pfad_old)
            f=open(pfad, 'w+')
            printf(("File ", lossfilename,  " renamed to "), lossfilename_old, printlog, logpath)
            printf(("File ", lossfilename,  " created. "), printlog, logpath)
    else:
        f=open(pfad, 'w')
        printf(("File ", lossfilename,  " created."), printlog, logpath)
    f.write('** Loss Test\n')
    for row in loss_phase['Test']:
        np.savetxt(f, np.array([str(float(row))]), fmt="%s")
        printf(np.array([str(float(row))]), printlog, logpath)
    f.close()


def traintest(model, bestmodel, optimizer, criterion, num_epochs, num_batches, loss_phase, best_loss, num_best_batch):
    model.train()
    best_model_wts = deepcopy(model.state_dict())
    # iter_train = iter(dataloaders['Training'])
    for epoch in range(epoch_start, num_epochs): # loop over the dataset multiple times
        iter_train = iter(dataloaders['Training'])
        for i in range(i_start, len(dataloaders['Training'])):
        # try:
            current_batch = i+epoch*len(dataloaders['Training'])
            printf('Train model ...', printlog, logpath)
            printf('Batch {}'.format(current_batch), printlog, logpath)
            try:
                data = next(iter_train)
            except FileNotFoundError as e:
                printf(e, printlog, logpath)
                # continue
            # except Exception as e:
            #     printf('Unknown error while iterating through data!', printlog, logpath)
            #     printf(sys.exc_info()[0], printlog, logpath)
            #     printf(e.__doc__, printlog, logpath)
            #     printf(e, printlog, logpath)
            #     continue
            # except Exception as e:
            #     printf(e, printlog, logpath)
            else:
                phase = 'Training'
                with torch.set_grad_enabled(phase == 'Training'):
                    inputs = data[0]['image'].to(device)
                    # inputs=inputs.double().to(device)
                    labels = data[0]['labels'].to(device)
                    labels = labels.float().to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    output = model(inputs)
                    # output = output.double()
                    # compute loss
                    loss = criterion['Training'](output, labels)
                    # loss = loss.double()
                    #print output and compare to labels
                    output = output.squeeze(0)
                # labels = labels.squeeze(0)
                printf('Backpropagation ...', printlog, logpath)
                loss.backward()
                optimizer.step()
                #
                #
                printf('Weights updated ...', printlog, logpath)
                labels=labels.cpu()
                for j in range(len(labels)):
                    output_history_h.append(output[j][0])
                    output_history_v.append(output[j][0])
                    label_history_h.append(labels[j][0])
                    label_history_v.append(labels[j][1])
                # print(label_history_h.shape())
                mean_h.append(np.mean(label_history_h))
                mean_v.append(np.mean(label_history_v))
                # dev_hor, dev_ver = print_results(i, output, labels, criterion)
            # append loss
                loss_phase['Training'].append(loss.detach())
                printf('Training loss: {}'.format(loss_phase['Training'][-1]), printlog, logpath)
                # Test
                if current_batch % 50 == 0 and current_batch != 0:
                # if current_batch % 1 == 0:
                    printf('Test model ...', printlog, logpath)
                    loss_phase, best_loss, num_best_batch, best_model_wts = test(model, criterion, current_batch, best_loss, loss_phase, num_best_batch, best_model_wts)
                if current_batch % 10 == 0 and current_batch != 0:
                    torch.save({
                        'epoch': epoch,
                        'batch_number': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_phase': loss_phase,
                        'label_history_h': label_history_h,
                        'label_history_v': label_history_v,
                        'output_history_h': output_history_h,
                        'output_history_v': output_history_v,
                        'mean_h': mean_h,
                        'mean_v': mean_v,
                        'best_loss': best_loss,
                        'num_best_batch': num_best_batch
                        }, checkpoint_path)
    # load best model state
    bestmodel.load_state_dict(best_model_wts)
    # compute testing loss of best model, and predictions and labels
    # loss_test, loss_vertical, loss_horizontal = testonly(model, criterion)
    printf('Final training loss: {}'.format(loss_phase['Training'][-1].item()), printlog, logpath)
    return model, bestmodel, loss_phase, label_history_h, label_history_v, mean_h, mean_v, num_best_batch, best_loss, output_history_h, output_history_v

def test(model, criterion, current_batch, best_loss, loss_phase, num_best_batch, best_model_wts):
    model.eval()
    batch_loss=0.0
    iter_test = iter(dataloaders['Test'])
    with torch.no_grad():
        for i in range(len(dataloaders['Test'])):
            try:
                data = next(iter_test)
            except FileNotFoundError as e:
                printf(e, printlog, logpath)
            else:
                try:
                    with torch.no_grad():
                        inputs = data[0]['image'].to(device)
                        labels = data[0]['labels'].to(device)
                        labels = labels.float().to(device)
                        output = model(inputs)
                        # compute loss
                        loss = criterion['Test'](output, labels)
                        #print output and compare to labels
                        output = output.squeeze(0)
                        labels = labels.squeeze(0)
                        batch_loss += loss.detach()
                except FileNotFoundError as e:
                    printf(e, printlog, logpath)
        # append loss
        loss_phase['Test'].append(batch_loss/len(dataloaders['Test']))
        printf('Current Testing loss: {}, best loss: {}'.format(loss_phase['Test'][-1].item(), best_loss), printlog, logpath)

        if loss_phase['Test'][-1].item() < best_loss:
            best_loss = loss_phase['Test'][-1].item()
            num_best_batch = current_batch
            best_model_wts = deepcopy(model.state_dict())
            printf('Model updated after batch: {}'.format(current_batch), printlog, logpath)
    return loss_phase, best_loss, num_best_batch, best_model_wts


def testonly(model, criterion, loss_horizontal, loss_vertical):
    model.eval()
    with torch.no_grad():
        iter_test = iter(dataloaders['Test'])
        loss_test = 0
        for i in range(len(dataloaders['Test'])):
            try:
                data = next(iter_test)
            except FileNotFoundError as e:
                printf(e, printlog, logpath)
            else:
                inputs, labels = data[0]['image'], data[0]['labels']
                labels = labels.float()
                output = model(inputs)
                # compute loss
                loss = criterion['Test'](output, labels)
                #print output and compare to labels
                output = output.squeeze(0)
                labels = labels.squeeze(0)
                dev_hor, dev_ver = print_results(i, output, labels, criterion)
                for j, out in enumerate(output):
                    loss_horizontal.append(np.array([out[0].item(),labels[j,0].item()]))
                    loss_vertical.append(np.array([out[1].item(),labels[j,1].item()]))
                # append loss
                loss_test += loss.detach()
    loss_test = loss_test/len(dataloaders['Test'])
    return loss_test, loss_vertical, loss_horizontal


class Net(nn.Module):
    def __init__(self, input_shape=(3,384,384)):
        super(Net, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18 = resnet18(pretrained=False)
        # self.resnet18.fc = nn.Linear(512,2)
        self.conv1 = nn.Conv2d(3,6,15, stride = 3, padding=7)
        self.bn1 = nn.BatchNorm2d(6)
        # self.do3 = nn.Dropout(p=0.1, inplace=False)
        self.conv2 = nn.Conv2d(6, 12, 11, stride = 1, padding=5)
        self.bn2 = nn.BatchNorm2d(12)
        # self.do4 = nn.Dropout(p=0.2, inplace=False)
        self.conv3 = nn.Conv2d(12, 24, 7, stride = 1, padding=3)
        self.bn3 = nn.BatchNorm2d(24)
        # self.do5 = nn.Dropout(p=0.2, inplace=False)
        self.conv4 = nn.Conv2d(24, 48, 3, stride = 1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        # self.do6 = nn.Dropout(p=0.2, inplace=False)
        # self.conv5 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        # self.bn5 = nn.BatchNorm2d(24)
        # self.do6 = nn.Dropout(p=0.8, inplace=False)
        # self.conv6 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        # self.bn6 = nn.BatchNorm2d(24)
        # self.do7 = nn.Dropout(p=0.8, inplace=False)
        n_size= self._get_conv_output(input_shape, batch_size['Training'])
        self.do1 = nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(n_size, 128) # from image dimension
        # self.do2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 2)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape, batch_size):
        bs = batch_size
        inputs = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(inputs)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        # x = F.max_pool2d(self.do3(F.relu(self.bn1(self.conv1(x)))), 2)
        # x = F.max_pool2d(self.do4(F.relu(self.bn2(self.conv2(x)))), 2)
        # x = F.max_pool2d(self.do5(F.relu(self.bn3(self.conv3(x)))), 2)
        # x = F.max_pool2d(self.do6(F.relu(self.bn4(self.conv4(x)))), 2)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        return x

    def forward(self, x):
        # x = self.resnet18(x)
        # Max pooling over a (2, 2) window
        x = self._forward_features(x)
        #
        x = x.view(x.size(0), -1) # results in vector (1xn tensor)
        x = self.do1(F.relu(self.fc1(x)))
        # x = self.do2(F.relu(self.fc2(x)))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class CompositeDataset(Dataset):
    """ Composite PCM images dataset."""
    def __init__(self, csv_file, simdir, imgpth, transform=None, phase='Training'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            train_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.simdir = simdir
        self.imgpth = imgpth
        self.transform = transform
        self.phase = phase
        self.results = pd.read_csv(os.path.join(simdir, phase, csv_file))
        # results = pd.read_csv(os.path.join(train_dir, 'resultsAll.dat'))
    def __len__(self):
        return len(self.results)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.imgpth, 'image_'+str(self.results.iloc[idx, 0])+'.png')
        # img_name = os.path.join(train_dir, 'image', 'image_'+str(results.iloc[0, 0])+'.png')
        try:
            image = Image.open(img_name, mode='r') # module PIL
            pixels = np.array(image)
            pixels = pixels[:442,:442,:3]
            image = Image.fromarray(pixels)
            maxsize=(384,384)
            image.thumbnail(maxsize, Image.ANTIALIAS)
            pixels = np.array(image)
            # convert from integers to floats
            pixels = pixels.astype('float32')
            # normalize to the range 0-1
            # print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
            pixels /= 255.0
            # confirm the normalization
            # print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
            # calculate per-channel means and standard deviations
            means = pixels.mean(axis=(0,1), dtype='float64')
            stds = pixels.std(axis=(0,1), dtype='float64')
            # print('Means: %s, Stds: %s' % (means, stds))
            # print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
            # per-channel centering of pixels
            pixels = (pixels - means) / stds
            # confirm it had the desired effect
            # means = pixels.mean(axis=(0,1), dtype='float64')
            # stds = pixels.std(axis=(0,1), dtype='float64')
            pixels = pixels.astype('float32')
            image = torch.tensor(pixels)
            image = torch.transpose(image,2,0)
            image = torch.transpose(image,2,1)
            image.type()
            results = self.results.iloc[idx, 2:4]
            # results = results.iloc[0,2:4]
            labels = pd.Series.to_numpy(results)
            sample = {'image': image, 'labels': labels}
            return sample, img_name
        except FileNotFoundError as e:
            printf(e, printlog, logpath)
            return img_name


#%% Print comment at top of log-file
for i, statement in enumerate(comment):
    printf(statement, printlog, logpath)

#%% Create Model
printf('Create model ...', printlog, logpath)
model = Net()
bestmodel = Net()
model.to(device) # .to(device) is used in case GPU can be used.
optimizer = optim.AdamW(model.parameters(), lr=lr) # default parameters used
criterion = {'Training': nn.MSELoss(), 'Test': nn.MSELoss()} # Loss functions for error estimation

printf("Model's state_dict:", printlog, logpath)
printf(model, printlog, logpath)        # print net state and layers used
printf(optimizer, printlog, logpath)    # print optimizer state and parameters

# load trained model (if testing or extending training)
# disclaimer:   Resume training when a run was aborted didn't work so far.
#               The model and optimizer are loaded and training is executed,
#               but the loss is again as high as at the beginning of training.
if load_model_bool == True:
    printf('Load model ...', printlog, logpath)
    if load_cancelled_model == True:        # If in the same folder, the previous run was cancelled
        checkpoint = torch.load(checkpoint_path)
        epoch_start = checkpoint['epoch']
        i_start = checkpoint['batch_number']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_phase = checkpoint['loss_phase']
        label_history_h = checkpoint['label_history_h']
        label_history_v = checkpoint['label_history_v']
        output_history_h = checkpoint['output_history_h']
        output_history_v = checkpoint['output_history_v']
        mean_h = checkpoint['mean_h']
        mean_v = checkpoint['mean_v']
        best_loss = checkpoint['best_loss']
        num_best_batch = checkpoint['num_best_batch']
    else:   # when model is tested, the final model is loaded
        final_checkpoint = torch.load(model_path)
        model.load_state_dict(final_checkpoint['model_state_dict'])
        optimizer.load_state_dict(final_checkpoint['optimizer_state_dict'])
        #
    # when training of a model should be extended: (doesn't work)
    if train_bool==True and load_model_bool==True and load_cancelled_model == False:
        checkpoint_objects = torch.load(objects_path)
        loss_phase = checkpoint_objects['loss_phase']
        loss_test = checkpoint_objects['loss_test']
        loss_horizontal = checkpoint_objects['loss_horizontal']
        loss_vertical = checkpoint_objects['loss_vertical']
        currentsim = checkpoint_objects['currentsim']
        num_best_batch = checkpoint_objects['num_best_batch']
        model_path = checkpoint_objects['model_path']
        label_history_h = checkpoint_objects['label_history_h']
        label_history_v = checkpoint_objects['label_history_v']
        output_history_h = checkpoint_objects['output_history_h']
        output_history_v = checkpoint_objects['output_history_v']
        mean_h = checkpoint_objects['mean_h']
        mean_v = checkpoint_objects['mean_v']
        best_loss = checkpoint_objects['best_loss']

# after loading parameters, model and optimizer have to be sent to device again
model = model.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# define characteristics
params = list(model.parameters())
params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
printf('The model has {} weights.'.format(params_count), printlog, logpath) # count model weights
#%% Start Program

# create datasets
printf('Create datasets ...', printlog, logpath)

dataset = {phase: CompositeDataset(csv_file='resultsAll.dat', simdir=simdir, imgpth=os.path.join(simdir, phase, imagepth), transform=None, phase=phase) for phase in ['Training', 'Test']}

#%% make and plot histogram of thermal conductivities, make weighted random samples

dataset_size = len(dataset['Training'])
printf('The dataset used for training has a total size of {} datapoints with 10% of its size as test data'.format(dataset_size), printlog, logpath)
num_batches = math.ceil(dataset_size/batch_size['Training'])

sampler = {'Training': None, 'Test': None}

# load dataset into loader
dataloaders = {
    'Training': DataLoader(dataset['Training'], batch_size=batch_size['Training'], shuffle=True, sampler=sampler['Training'], num_workers=2), 'Test': DataLoader(dataset['Test'], batch_size=batch_size['Test'], shuffle=False, sampler=sampler['Test'], num_workers=2)}

#%% Training and Testing
if train_bool==True:
    if load_model_bool == False:
        loss_horizontal = []
        loss_vertical = []
        label_history_h = []
        label_history_v = []
        output_history_h = []
        output_history_v = []
        mean_h = []
        mean_v = []
        loss_phase = {'Training': [], 'Test': []}
        num_best_batch = 0
        best_loss = 10000000000.0
    if load_cancelled_model == False:
        epoch_start = 0
        i_start = 0
    model, bestmodel, loss_phase, label_history_h, label_history_v, mean_h, mean_v, num_best_batch, best_loss, output_history_h, output_history_v = traintest(model, bestmodel, optimizer, criterion, num_epochs, num_batches, loss_phase, best_loss, num_best_batch)
    printf('Testing model {}...'.format(model_name), printlog, testlogpath)
    ## save best test loss batch
    savelosshistoryfiles(loss_phase, currentsim) # .txt
    ## save final model and best model
    if save_model_bool == True:
        printf('Save model ... {}'.format(model_name), printlog, logpath)
        model_path = os.path.join(currentsim, model_name)
        bestmodel_path = os.path.join(currentsim, bestmodel_name)
        #
    loss_test = 0.0
    loss_vertical = []
    loss_horizontal = []
    loss_test, loss_vertical, loss_horizontal = testonly(model, criterion, loss_horizontal, loss_vertical)
    printf('Final testing loss {} after {} batches'.format(loss_test, num_best_batch), printlog, logpath)
    printf('\nHorizontal predictions and labels', printlog, testlogpath)
    for j, out_h in enumerate(loss_horizontal):
        printf(str(out_h[0])+', '+str(out_h[1]), printlog, testlogpath)
    printf('\nVertical predictions and labels', printlog, testlogpath)
    for j, out_v in enumerate(loss_vertical):
        printf(str(out_v[0])+', '+str(out_v[1]), printlog, testlogpath)
        #
# test loop: only if the network is not trained, just tested
if test_bool == True and train_bool == False:
    printf('Testing model {}...'.format(model_name), printlog, testlogpath)
    loss_test = 0.0
    loss_horizontal = []
    loss_vertical = []
    loss_test, loss_vertical, loss_horizontal = testonly(model, criterion, loss_horizontal, loss_vertical)
    # optional: save to file(if only testing: dirName has to be set manually to existing directory)
    printf('\nHorizontal predictions and labels', printlog, testlogpath)
    for j, out_h in enumerate(loss_horizontal):
        printf(str(out_h[0])+', '+str(out_h[1]), printlog, testlogpath)
    printf('\nVertical predictions and labels', printlog, testlogpath)
    for j, out_v in enumerate(loss_vertical):
        printf(str(out_v[0])+', '+str(out_v[1]), printlog, testlogpath)

# Saving the objects:
if save_model_bool == True:
    printf('Save model ... {}'.format(model_name), printlog, logpath)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
torch.save({
    'loss_phase': loss_phase,
    'loss_test': loss_test,
    'loss_horizontal': loss_horizontal,
    'loss_vertical': loss_vertical,
    'currentsim': currentsim,
    'num_best_batch': num_best_batch,
    'model_path': model_path,
    'label_history_h': label_history_h,
    'label_history_v': label_history_v,
    'output_history_h': output_history_h,
    'output_history_v': output_history_v,
    'mean_h': mean_h,
    'mean_v': mean_v,
    'best_loss': best_loss,
    }, objects_path)

printf('Program end: {}'.format(datetime.datetime.now()), printlog, logpath)