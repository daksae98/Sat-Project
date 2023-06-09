import os
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from PIL import Image
import math
import random


import numpy as np

# import matplotlib.pyplot as plt

class WildFireDataset(Dataset):
    """Korea Wild Fire Dataset"""
    def __init__(self,data, transform=None):
        """
        Arguments:
            csv_file (string) : Path to the csv file with annotations
            transform (callable, optional) : Optinal transform to be applied
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample = self.data[idx]
        images = sample[:-1]
        labels = sample[-1]
        
        if self.transform:
            images = self.transform(images)
    
        return (images,labels)
        
## Transforms
class ToTensor(object):
    def __call__(self,image):        
        return torch.from_numpy(image)

class RandomRotate(object):
    def __init__(self,p,max_angle):
        '''p = 0~1, max_angle : degree(float)'''
        assert isinstance(p,float),'p should be int'
        self.p = p
        assert isinstance(max_angle,float),'angle should be float'
        self.max_angle = max_angle

    def __call__(self,image):
        r = random.random()
        angle = math.ceil(self.max_angle * r)
        if self.p >= r:
            image_tensor = torch.from_numpy(image)
            rotated_tensor = []
            for channel in image_tensor:
                rotated_channel = F.rotate(channel.unsqueeze(0), angle)
                rotated_tensor.append(rotated_channel.squeeze(0))
                
            rotated_image = torch.stack(rotated_tensor, dim=0).numpy()

            return rotated_image
        
        else:
            return image




#(80, 6, 128, 128)
def get_train_test_dataloader(train_npys,test_npys):
    '''데이터 자체를 중복되게 잘랐기 때문에, random_split 했을때, train과 test간 겹치는게 존재함. 따라서 sokcho랑 donghae를 validation set으로 사용.'''
    
    train_data = []
    for npy in train_npys:
        item = np.load(npy)
        item = item.tolist()
        train_data = train_data + item      
    train_data = np.array(train_data)

    test_data = []
    for npy in test_npys:
        item = np.load(npy)
        item = item.tolist()
        test_data = test_data + item      
    test_data = np.array(test_data)
    

    # transform train
    transform_train = WildFireDataset(train_data,transform=transforms.Compose([
        # RandomRotate(p=0.5,max_angle=90.0),
        ToTensor()
    ]))
   
    # transform test
    transform_test = WildFireDataset(test_data, transform=transforms.Compose([
        ToTensor()
    ]))

    train_dataloader = DataLoader(transform_train,batch_size=16,shuffle=True,num_workers=2)
    test_dataloader = DataLoader(transform_test,batch_size=16,shuffle=True,num_workers=2)

    return train_dataloader, test_dataloader


