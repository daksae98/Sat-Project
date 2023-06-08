import matplotlib.pyplot as plt
from PIL import Image
import os


images = [
# '20_0_12_pred_361.png', '20_0_12_true_361.png',
# '30_0_12_pred_357.png', '30_0_12_true_357.png',
# '40_0_12_pred_413.png', '40_0_12_true_413.png',
# '50_0_12_pred_414.png', '50_0_12_true_414.png',
# '60_0_12_pred_397.png', '60_0_12_true_397.png',
# '70_0_12_pred_390.png', '70_0_12_true_390.png',
# '80_0_12_pred_428.png', '80_0_12_true_428.png',
# '90_0_12_pred_426.png', '90_0_12_true_426.png',
# '100_0_12_pred_418.png', '100_0_12_true_418.png', 
# '110_0_12_pred_424.png', '110_0_12_true_424.png',
# '120_0_12_pred_442.png', '120_0_12_true_442.png', 
# '130_0_12_pred_484.png', '130_0_12_true_484.png',
# '140_0_12_pred_503.png', '140_0_12_true_503.png',
# '150_0_12_pred_508.png', '150_0_12_true_508.png',
('160_0_12_pred_516.png', '160_0_12_true_516.png'),
('170_0_12_pred_522.png', '170_0_12_true_522.png'),
('180_0_12_pred_520.png', '180_0_12_true_520.png'),
('190_0_12_pred_508.png', '190_0_12_true_508.png'),
('200_0_12_pred_534.png', '200_0_12_true_534.png'),
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
