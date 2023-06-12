from unet import UNet
import torch
import numpy as np
import os
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# 훈련시킨 모델을 불러와서, dNBR 계산 결과를 데이터셋별로 하나씩 시도. 
# 이미지랑 Npy로 저장. [pred, true_dNBR, dNDVI] plot


Uljin1 = np.load('npys/valid/Uljin1_8.npy')


Uljin1_dNDVI = np.load('npys/evaluation/Uljin1_dNDVI_8.npy')



def forward():
    '''pred true_dNBR, true_dNDVI'''
    stack = []
    net.eval()
    # for batch_idx,(image, true_mask) in enumerate(test_dataloader):
    #     pred = net(image)
    
    for idx,image in enumerate(Uljin1):
        # ToTensor
        image = torch.from_numpy(image)
        
        # Feature, True Mask, dNDVI
        feature = image[:-1].unsqueeze(0)
        true_dNBR = image[-1].detach().numpy()
        true_dNDVI = Uljin1_dNDVI[idx]
        
        # Forward
        pred = net(feature)
        pred = pred.squeeze(0)
        pred_numpy = pred.detach().numpy()
        pred_numpy = np.argmax(pred_numpy, axis=0)
        stack.append([pred_numpy,true_dNBR,true_dNDVI])

    # print(stack)
    res = np.array(stack)
    np.save('Uljin1',res)
    
        

def test():
    H = np.load('Uljin1.npy')
    print(H.shape)
    for pred, dNBr, dNDVI in H:
        fig,axs = plt.subplot(3,1)
        axs[0].imshow(pred)
        axs[1].imshow(dNBr)
        axs[2].imshow(dNDVI)
        plt.show()


if __name__ == '__main__':
    # test()
    
    n_classes = 8
    device = 'cpu'
    net = UNet(n_channels=5,n_classes=n_classes,bilinear=False)
    net.load_state_dict(torch.load('net_checkpoint/unet_300_563', map_location=device))
    
    
    # TRAIN_DIR = 'npys/train/'
    # VALID_DIR = 'npys/valid/'
    # train_npys = [TRAIN_DIR + path for path in os.listdir(TRAIN_DIR)]
    # test_npys = [VALID_DIR + path for path in os.listdir(VALID_DIR)]
    # train_dataloader, test_dataloader = get_train_test_dataloader(train_npys,test_npys)
    
    forward()
    