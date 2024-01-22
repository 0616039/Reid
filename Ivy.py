# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image


#######################################################################
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_image_dir',default='../Market/pytorch',type=str, help='./test_img')
parser.add_argument('--support_image_dir',default='../Market/pytorch',type=str, help='./support_img')
parser.add_argument('--test_ID',default='-1', type=int, help='test_image ID' )
parser.add_argument('--support_ID',default='-1', type=int, help='support_image ID' )
parser.add_argument('--extract_test',default='-1', type=int, help='only extract: 1' )
parser.add_argument('--remove_add',default='-1', type=int, help='remove: 1 add: 0' )
parser.add_argument('--remove_ID',default='-1', type=int, help='remove ID' )
parser.add_argument('--rank',default='-1', type=int, help='-1: no need to rank, > 0: output k rank' )
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')

opt = parser.parse_args()
###load config###
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.stride = config['stride']
opt.nclasses = config['nclasses']
opt.ibn = config['ibn']
opt.linear_num = config['linear_num']
        
######################################################################
# Load Data
# ---------
class SingleImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return image
        
data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model', opt.name,'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, image):
    since = time.time()
    with torch.no_grad():
        if opt.linear_num <= 0:
            opt.linear_num = 2048
        single_image_dataset = SingleImageDataset(image, transform=data_transforms)
        dataloader = DataLoader(single_image_dataset, batch_size=32, shuffle=False, num_workers=16)
        
        for iter, img in enumerate(dataloader):
            #img, label = data 
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
        
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                outputs = model(input_img) 
                ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            if iter == 0:
                features = torch.FloatTensor( len(dataloader.dataset), ff.shape[1])
                #features = torch.cat((features,ff.data.cpu()), 0)
            start = iter*32
            end = min( (iter+1)*32, len(dataloader.dataset))
            features[ start:end, :] = ff
        
    time_elapsed = time.time() - since
    print('Extract complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    return features

def remove_id(data, ID):
    labels = data['gallery_label']
    
    indices_to_remove = [i for i, label in enumerate(labels[0]) if label == ID]
    # print(indices_to_remove)

    new_data = {}
    new_data['gallery_f'] = data['gallery_f'][[i for i in range(len(data['gallery_f'])) if i not in indices_to_remove]]
    new_data['gallery_label'] = [data['gallery_label'][0][[i for i in range(len(data['gallery_label'][0])) if i not in indices_to_remove]]]
    
    return new_data

#######################################################################
# sort the images
def sort_img(qf, ql, gf, gl, rank):
    since = time.time()
    query = qf.view(-1,1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    
    # good index
    query_index = np.argwhere(gl==ql)

    junk_index1 = np.argwhere(gl==-1)

    mask = np.in1d(index, junk_index1, invert=True)
    index = index[mask]
    
    time_elapsed = time.time() - since
    print('Sort complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    
    
    print('Top ', rank, ' images are as follow:')
    try:
        for i in range(rank):
            label = gl[index[i]]
            print('Top: ', i+1, ' :', label)
    except RuntimeError:
        for i in range(rank):
            label = gl[index[i]]
            print('Top: ', i+1, ' :', label)
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    
    return index
#######################################################################

if __name__ == '__main__':
    #### model start ####
    print('-------Model-----------')
    model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)
    
    model = load_network(model_structure)
    model.classifier.classifier = nn.Sequential()

    model = model.eval()
    if use_gpu:
        model = model.cuda()
    
    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)
    
    print(model)
    #### model end ####
    images = []
    image = Image.open(opt.test_image_dir)
    images.append(image)
    for i in range(31):
        images.append(image)
    global Support_data
    Support_data = {}
    # Support_data = scipy.io.loadmat('support_image_result.mat')
    
    # Extract feature
    if opt.extract_test == 1:
        query_feature = extract_feature(model, images)
        print(query_feature.size())
        image = Image.open(opt.test_image_dir)
        images = []
        images.append(image)
        query_feature = extract_feature(model, images)
        images = []
        images.append(image)
        query_feature = extract_feature(model, images)
        images = []
        images.append(image)
        query_feature = extract_feature(model, images)
        images = []
        images.append(image)
        query_feature = extract_feature(model, images)    
    
    # Modify Suppoort set
    if opt.remove_add == 0:
        # Extract feature
        new_support_feature = extract_feature(model, image)
        
        if bool(Support_data):
            print("It is not empty")
            support_feature = torch.FloatTensor(Support_data['gallery_f'])
            support_label = Support_data['gallery_label'][0]
            
            support_feature = torch.cat((support_feature, new_support_feature), dim=0)
            support_label = np.append(support_label, opt.support_ID)
            
            result = {'gallery_f': support_feature.numpy(), 'gallery_label': opt.support_ID}
                
        else:
            print("It is empty")
            Support_data = {'gallery_f': new_support_feature.numpy(), 'gallery_label': opt.support_ID}
        
      
    elif opt.remove_add == 1:
        Support_data = remove_id(Support_data, opt.remove_ID)    #change to support set
        print(len(Support_data['gallery_label'][0]))
    
    # Output Rank
    if opt.rank > 0:
        query_feature = extract_feature(model, image)
        support_feature = torch.FloatTensor(Support_data['gallery_f'])
        support_label = Support_data['gallery_label'][0]
        rank = min(opt.rank, support_feature.size()[0])
        print(support_feature.size())

        index = sort_img(query_feature[0], opt.test_ID, support_feature, support_label, rank)
