import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import os


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
