import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from tqdm import tqdm
# from resnet import resnet18
from unet import UNet
from custom_dataloader import get_train_test_dataloader
from utils.dice_score import dice_loss,multiclass_dice_coeff,dice_coeff
from PIL import Image

npys = ['npys/Anndong_32.npy','npys/Donghae_8.npy','npys/Sokcho_8.npy','npys/Uljin1_8.npy','npys/Uljin3_24.npy']

def train(epoch):
    net.train()
    epoch_loss = 0
    global_step = 0
    for batch_idx,(images, true_masks) in enumerate(train_dataloader):
        print(epoch,batch_idx)
        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device = device, dtype = torch.float32,memory_format=torch.channels_last)
        true_masks = true_masks.to(device = device, dtype = torch.long)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
            masks_pred = net(images)
            if net.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss += dice_loss(F.sigmoid())
            else:
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, net.n_classes).permute(0,3,1,2).float(),
                    multiclass=True
                )

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        global_step += 1
        epoch_loss += loss.item()
        
@torch.inference_mode()
def eval(epoch):
    net.eval()
    num_val_batches = len(test_dataloader)
    dice_score = 0
    

    for batch_idx,(image, true_mask) in enumerate(test_dataloader):
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
            image = image.to(device = device, dtype=torch.float32, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.long)

            mask_pred = net(image)
           
        #    pred_image = Image.fromarray(mask_pred)
        #    pred_image.save(f'res/{epoch}_{batch_idx}_dNBR.tif')
            assert true_mask.min() >= 0 and true_mask.max() < net.n_classes, 'True mask indices should be in [0,n_classes]'
           # convert to one-hot format
            true_mask = F.one_hot(true_mask, net.n_classes).permute(0,3,1,2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0,3,1,2).float()
           
            pred_images = np.argmax(mask_pred, axis=1)
            for idx,pred_image in enumerate(pred_images): 
                pred_image = np.array(pred_image,dtype=np.uint8)
                pred_image = Image.fromarray(pred_image)
                pred_image.save(f'res/{epoch}_{batch_idx}_{idx}_pred.png')
           
            true_images = np.argmax(true_mask, axis=1)
            for idx,true_image in enumerate(true_images):
                true_image = np.array(true_image,dtype=np.uint8)
                true_image = Image.fromarray(true_image)
                true_image.save(f'res/{epoch}_{batch_idx}_{idx}_true.png')
           
           
           #compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:,1:], true_mask[:,1:], reduce_batch_first=False)
    
    net.train()

    return dice_score / max(num_val_batches, 1)
    
        
'''
epochs: int = 5,
batch_size: int = 1,
learning_rate: float = 1e-5,
val_percent: float = 0.1,
save_checkpoint: bool = True,
img_scale: float = 0.5,
amp: bool = False,
weight_decay: float = 1e-8,
momentum: float = 0.999,
gradient_clipping: float = 1.0,
'''

if __name__ == '__main__':
    
    epochs = 10

    net = UNet(n_channels=5,n_classes=7,bilinear=False)
    net = net.to(memory_format=torch.channels_last) # beta

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if torch.backends.mps.is_available():
    #     mps_device = torch.device("mps")
    #     net.to(mps_device)
    net.to(device)
    oss_function = nn.CrossEntropyLoss()
    train_dataloader, test_dataloader = get_train_test_dataloader(npys, test_size=0.2)
    
    optimizer = optim.RMSprop(net.parameters(), lr=1e-5, weight_decay=1e-8,momentum=0.999,foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        train(epoch)
        val_score = eval(epoch)
        print(f'#{epoch} val_score:{val_score}')
