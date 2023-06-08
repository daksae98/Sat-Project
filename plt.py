import matplotlib.pyplot as plt
from PIL import Image


fig, axs = plt.subplots(4, 2)

for i in range(4):
    pred_image = Image.open(f'res/6_0_{i+8}_pred.png')
    true_image = Image.open(f'res/6_0_{i+8}_true.png')
    axs[i][0].imshow(true_image)
    axs[i][1].imshow(pred_image)

plt.show()