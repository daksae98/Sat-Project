import numpy as np
import math
from PIL import Image
from create_dataset import load_images

# dNDVI load...


def cal_dNDVI(prev, post, region):
    '''
    dNDVI = NDVI_prev - NDVI_post
    NDVI = (NIR - R)/(NIR + R)
    '''
    [_, _, prev_r, prev_nir, _] = prev
    [_, _, post_r, post_nir, _] = post

    # dNDVI
    prev_sum = prev_nir + prev_r
    prev_dif = prev_nir - prev_r
    post_sum = post_nir + post_r
    post_dif = post_nir - post_r

    prev_0_idxs = np.where(prev_sum == 0)
    prev_sum[prev_0_idxs] = 1
    prev_dif[prev_0_idxs] = 0

    post_0_idxs = np.where(post_sum == 0)
    post_sum[post_0_idxs] = 1
    post_dif[post_0_idxs] = 0

    prev_NDVI = prev_dif/prev_sum
    post_NDVI = post_dif/post_sum
    dNDVI = prev_NDVI - post_NDVI

    im_dNDVI = Image.fromarray(dNDVI)
    im_dNDVI.save(f'Datasets/{region}/{region}_dNDVI.tif')

    return dNDVI

def crop_128_valid(npy,region):

    row, col = npy.shape
    ir, ic = math.ceil(row/256), math.ceil(col/256)

    res = []
    # 좌상단 시작
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[r*128:(r+1)*128, c*128:(c+1)*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 좌 하단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[ row - (r+1)*128:row - (r)*128, c*128:(c+1)*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 우상단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[r*128:(r+1)*128, col-(c+1)*128:col-c*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 우하단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[row - (r+1)*128:row - (r)*128,
                       col-(c+1)*128:col-c*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1

    res = np.array(res)

    for idx,croped_image in enumerate(res):
        im_dNDVI = Image.fromarray(croped_image)
        im_dNDVI.save(f'Datasets/evaluation/{region}/{region}_{idx}_dNDVI.tif')
    np.save(f'npys/evaluation/{region}_dNDVI_{res.shape[0]}', res)


def cal_correlation_coef():
    print()

if __name__ == '__main__':
    REGION = 'Uljin1'
    # img = load_image('Datasets/Uljin1/Uljin1_dNBR_label.tif')
    # plot_image(img)
    prev, post = load_images(REGION)
    dNDVI = cal_dNDVI(prev, post, REGION)
    crop_128_valid(dNDVI,REGION)