import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix

'''
images = [
('400_1_7_true_476.png','400_1_7_pred_476.png'),
('400_1_7_true_439.png','400_1_7_pred_439.png')
]
cmap = mpl.colormaps['plasma']  # 'jet' 색상 맵을 사용하며, 8개의 색상으로 구성

# 이미지를 imshow로 표시

fig, axs = plt.subplots(2, 2)
dirs=os.listdir('res')
dirs.sort()
for i, (pred,true)in enumerate(images):

    pred_image = Image.open('res/'+pred)
    true_image = Image.open('res/'+true)
    axs[i][0].imshow(true_image, cmap=cmap, vmin=0, vmax=7)
    axs[i][1].imshow(pred_image, cmap=cmap, vmin=0, vmax=7)

plt.show()
'''
def test():
    cmap = mpl.colormaps['plasma']  # 'jet' 색상 맵을 사용하며, 8개의 색상으로 구성
    H = np.load('Hongsung.npy')
    print(H.shape)
    for idx,[pred, dNBr, dNDVI] in enumerate(H):
        fig,axs = plt.subplots(1,3)
        fig.set_size_inches(10,10)
        axs[0].imshow(pred,cmap=cmap, vmin=0, vmax=7)
        axs[0].set_title('Pred')
        axs[1].imshow(dNBr,cmap=cmap, vmin=0, vmax=7)
        axs[1].set_title('True')
        axs[2].imshow(dNDVI)
        axs[2].set_title('dNDVI')
        correlation = np.corrcoef(pred.flatten(), dNDVI.flatten())
        cf = confusion_matrix(pred.flatten(), dNBr.flatten())
        print(correlation)
        print(cf)
        plt.savefig(f'Hongsung_{idx}')
        plt.show()
        

test()