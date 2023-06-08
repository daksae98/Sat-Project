import os
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
from PIL import Image

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




# transformed_dataset = WildFireDataset(npys,transform=transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     # transforms.Normalize()
# ]))

# dataloader = DataLoader(transformed_dataset, batch_size=16,shuffle=True,num_workers=0)

# print(npys.shape)


#(80, 6, 128, 128)
def get_train_test_dataloader(npys,test_size = 0.2):
    # npys = ['npys/Anndong_32.npy','npys/Donghae_8.npy','npys/Sokcho_8.npy','npys/Uljin1_8.npy','npys/Uljin3_24.npy']
    data = []
    for npy in npys:
        item = np.load(npy)
        item = item.tolist()
        data = data + item      
    data = np.array(data)

    # test train split
    data_count = len(data)
    test_count = int(test_size*data_count)
    train_count = data_count - test_count
    train_data, test_data = random_split(data,[train_count,test_count])
    # transform train
    transform_train = WildFireDataset(train_data,transform=transforms.Compose([
        ToTensor()
    ]))
   
    # transform test
    transform_test = WildFireDataset(test_data, transform=transforms.Compose([
        ToTensor()
    ]))

    train_dataloader = DataLoader(transform_train,batch_size=16,shuffle=True,num_workers=2)
    test_dataloader = DataLoader(transform_test,batch_size=16,shuffle=True,num_workers=2)

    return train_dataloader, test_dataloader


