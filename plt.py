import matplotlib.pyplot as plt
from PIL import Image
import os


images = [
# ('200_0_12_pred_238.png', '200_0_12_true_238.png'),
# ('250_0_12_pred_234.png', '250_0_12_true_234.png'),
# ('300_0_12_pred_221.png', '300_0_12_true_221.png'),
# ('350_0_12_pred_207.png', '350_0_12_true_207.png'),
# ('400_0_0_pred_233.png', '400_0_0_true_233.png'),
# ('400_0_1_pred_233.png', '400_0_1_true_233.png'),
# ('400_0_2_pred_233.png', '400_0_2_true_233.png'),
# ('400_0_3_pred_233.png', '400_0_3_true_233.png'),
# ('400_0_4_pred_233.png', '400_0_4_true_233.png'),
# ('400_0_5_pred_233.png', '400_0_5_true_233.png'),
('400_0_6_pred_233.png', '400_0_6_true_233.png'),
('400_0_7_pred_233.png', '400_0_7_true_233.png'),
('400_0_8_pred_233.png', '400_0_8_true_233.png'),
('400_0_9_pred_233.png', '400_0_9_true_233.png'),
('400_0_11_pred_233.png', '400_0_11_true_233.png'),
# ('400_0_10_pred_233.png', '400_0_10_true_233.png'),
# ('400_0_12_pred_233.png', '400_0_12_true_233.png'),
# ('400_0_13_pred_233.png', '400_0_13_true_233.png'),
# ('400_0_14_pred_233.png', '400_0_14_true_233.png'),
# ('400_0_15_pred_233.png', '400_0_15_true_233.png'),
]

fig, axs = plt.subplots(5, 2)
dirs=os.listdir('res')
dirs.sort()
for i, (pred,true)in enumerate(images):

    pred_image = Image.open('res/'+pred)
    true_image = Image.open('res/'+true)
    axs[i][0].imshow(true_image)
    axs[i][1].imshow(pred_image)

plt.show()
