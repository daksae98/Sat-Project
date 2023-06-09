import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import os


images = [
# ('100_0_12_pred_460.png', '100_0_12_true_460.png'),
# ('150_0_12_pred_423.png', '150_0_12_true_423.png'),
# ('200_0_12_pred_454.png', '200_0_12_true_454.png'),
# ('250_0_12_pred_463.png', '250_0_12_true_463.png'),
# ('300_0_12_pred_481.png', '300_0_12_true_481.png'),
# ('350_0_12_pred_493.png', '350_0_12_true_493.png'),
# ('400_0_0_pred_503.png', '400_0_0_true_503.png'),
# ('400_0_10_pred_503.png', '400_0_10_true_503.png'),
# ('400_0_11_pred_503.png', '400_0_11_true_503.png'),
# ('400_0_12_pred_503.png', '400_0_12_true_503.png'),
# ('400_0_13_pred_503.png', '400_0_13_true_503.png'),
# ('400_0_14_pred_503.png', '400_0_14_true_503.png'),
# ('400_0_15_pred_503.png', '400_0_15_true_503.png'),
('400_0_1_pred_503.png', '400_0_1_true_503.png'),
('400_0_2_pred_503.png', '400_0_2_true_503.png'),
('400_0_3_pred_503.png', '400_0_3_true_503.png'),
('400_0_4_pred_503.png', '400_0_4_true_503.png'),
('400_0_5_pred_503.png', '400_0_5_true_503.png'),
# ('400_0_6_pred_503.png', '400_0_6_true_503.png'),
# ('400_0_7_pred_503.png', '400_0_7_true_503.png'),
# ('400_0_8_pred_503.png', '400_0_8_true_503.png'),
# ('400_0_9_pred_503.png', '400_0_9_true_503.png'),
# ('50_0_12_pred_359.png', '50_0_12_true_359.png')
]
cmap = mpl.colormaps['plasma']  # 'jet' 색상 맵을 사용하며, 8개의 색상으로 구성

# 이미지를 imshow로 표시

fig, axs = plt.subplots(5, 2)
dirs=os.listdir('res')
dirs.sort()
for i, (pred,true)in enumerate(images):

    pred_image = Image.open('res/'+pred)
    true_image = Image.open('res/'+true)
    axs[i][0].imshow(true_image, cmap=cmap, vmin=0, vmax=7)
    axs[i][1].imshow(pred_image, cmap=cmap, vmin=0, vmax=7)

plt.show()
