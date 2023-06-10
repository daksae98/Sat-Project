import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


# from resnet import resnet18
from unet import UNet
from custom_dataloader import get_train_test_dataloader
from utils.dice_score import dice_loss,multiclass_dice_coeff,dice_coeff
from PIL import Image

###
TRAIN_DIR = 'npys/train/'
VALID_DIR = 'npys/valid/'
###

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

            assert true_mask.min() >= 0 and true_mask.max() < net.n_classes, 'True mask indices should be in [0,n_classes]'
            # convert to one-hot format
            true_mask = F.one_hot(true_mask, net.n_classes).permute(0,3,1,2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0,3,1,2).float()
            
            
            #compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:,1:], true_mask[:,1:], reduce_batch_first=False)
            score = dice_score / max(num_val_batches,1)
            
            idx = 0
            
            if epoch % 50 == 0:
                score_cpu = score.cpu()
                score_cpu = np.array(score_cpu)
                score_round = int(round(score_cpu*1000,0))
                
                mask_pred_cpu = mask_pred.cpu()
                pred_images = np.argmax(mask_pred_cpu, axis=1)
                true_mask_cpu = true_mask.cpu()
                true_images = np.argmax(true_mask_cpu, axis=1)

                if epoch == epochs:
                    torch.save(net.state_dict(), f'net_checkpoint/unet_{epochs}')
                    for i, pred_image in enumerate(pred_images):
                        pred_image = np.array(pred_image,dtype=np.uint8)
                        pred_image = Image.fromarray(pred_image)
                        pred_image.save(f'res2/{epoch}_{batch_idx}_{i}_pred_{score_round}.png')
                    for i, true_image in enumerate(true_images):
                        true_image = np.array(true_image,dtype=np.uint8)
                        true_image = Image.fromarray(true_image)
                        true_image.save(f'res2/{epoch}_{batch_idx}_{i}_true_{score_round}.png')
                else:
                    pred_image = np.array(pred_images[idx],dtype=np.uint8)
                    pred_image = Image.fromarray(pred_image)
                    pred_image.save(f'res2/{epoch}_{batch_idx}_{idx}_pred_{score_round}.png')
                    
                    true_image = np.array(true_images[idx],dtype=np.uint8)
                    true_image = Image.fromarray(true_image)
                    true_image.save(f'res2/{epoch}_{batch_idx}_{idx}_true_{score_round}.png')
           
           
            
    
    net.train()

    return score
    

if __name__ == '__main__':
    
    epochs = 400
    water_labeled = True
    n_classes = 8 if water_labeled else 7

    net = UNet(n_channels=5,n_classes=n_classes,bilinear=False)
    net = net.to(memory_format=torch.channels_last) # beta

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.to(device)
    oss_function = nn.CrossEntropyLoss()

    train_npys = [TRAIN_DIR + path for path in os.listdir(TRAIN_DIR)]
    test_npys = [VALID_DIR + path for path in os.listdir(VALID_DIR)]
    train_dataloader, test_dataloader = get_train_test_dataloader(train_npys,test_npys)
    
    optimizer = optim.RMSprop(net.parameters(), lr=1e-5, weight_decay=1e-8,momentum=0.999,foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

   



    # for epoch in range(1, epochs + 1):
    #     train(epoch)
    #     val_score = eval(epoch)
    #     print(f'#{epoch} val_score:{val_score}')
