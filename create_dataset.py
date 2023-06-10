
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# 데이터 불러와서, DNBR 계산하고
# DNBR Label, DNBR 영상을 만든다
# 128 * 128로 나눈다.

REGION = 'Uljin3'

BANDS = [
    "B2_BLUE",
    "B3_GREEN",
    "B4_RED",
    "B8_NIR",
    "B11_SWIR",
]

SEVERITY_LEVEL = {
    'EnhancedRegrowthHigh': {
        'RANGE': [-0.500, -0.250],
        'VALUE': 0
    },
    'EnhancedRegrowthLow': {
        'RANGE': [-0.250, -0.100],
        'VALUE': 1
    },
    'Unburned': {
        'RANGE': [-0.100, 0.100],
        'VALUE': 2
    },
    'LowSeverity': {
        'RANGE': [0.100, 0.270],
        'VALUE': 3
    },
    'ModerateLowSeverity': {
        'RANGE': [0.270, 0.440],
        'VALUE': 4
    },
    'MiderateHighSeverity': {
        'RANGE': [0.440, 0.660],
        'VALUE': 5
    },
    'HighSeverity': {
        'RANGE': [0.660, 1.300],
        'VALUE': 6
    }
}


def load_image(path):
    img_ds = gdal.Open(path)
    img = img_ds.ReadAsArray()
    img_arr = np.array(img)
    return img_arr


def plot_image(img):
    plt.imshow(img)
    plt.show()


def load_images(region):
    prev_paths = [
        f'Datasets/{region}/{region}_{band}_prev.tif' for band in BANDS]
    post_paths = [
        f'Datasets/{region}/{region}_{band}_post.tif' for band in BANDS]
    prev = []
    post = []
    for path in prev_paths:
        img_ds = gdal.Open(path)
        img = img_ds.ReadAsArray()
        img_arr = np.array(img)
        _nan_idx = np.argwhere(img_arr == np.nan)
        img_arr[_nan_idx] = 0
        for idx in range(len(_nan_idx)):
            img_arr[_nan_idx[idx][0]][_nan_idx[idx][1]] = 0
        prev.append(img_arr)

    for path in post_paths:
        img_ds = gdal.Open(path)
        img = img_ds.ReadAsArray()
        img_arr = np.array(img)
        _nan_idx = np.argwhere(np.isnan(img_arr))
        for idx in range(len(_nan_idx)):
            img_arr[_nan_idx[idx][0]][_nan_idx[idx][1]] = 0
        post.append(img_arr)

    prev = np.array(prev)
    post = np.array(post)
    return prev, post




def cal_dNBR(prev, post, region):
    '''
    dNDVI = NDVI_prev - NDVI_post
    NDVI = (NIR - R)/(NIR + R)
    '''
    [_, _, prev_r, prev_nir, prev_swir] = prev
    [_, _, post_r, post_nir, post_swir] = post
    
    # dNBR
    prev_sum = prev_nir + prev_swir
    prev_dif = prev_nir - prev_swir
    post_sum = post_nir + post_swir
    post_dif = post_nir - post_swir

    prev_0_idxs = np.where(prev_sum == 0)
    prev_sum[prev_0_idxs] = 1
    prev_dif[prev_0_idxs] = 0

    post_0_idxs = np.where(post_sum == 0)
    post_sum[post_0_idxs] = 1
    post_dif[post_0_idxs] = 0

    prev_nbr = prev_dif/prev_sum
    post_nbr = post_dif/post_sum
    dnbr = prev_nbr - post_nbr

    # Save
    im_dnbr = Image.fromarray(dnbr)
    im_dnbr.save(f'Datasets/{region}/{region}_dNBR.tif')

    return dnbr


def label_dNBR(region,prev,label_water=False):
    def find_water(prev):
        [b_prev, g_prev, r_prev, prev_nir, prev_r] = prev

        prev_0_idx = np.where(prev_nir == 0)
        prev_nir[prev_0_idx] = 1

        water = b_prev/prev_nir
        water_idx = np.where(water > 1.3)
        water = np.zeros_like(b_prev)
        water[water_idx] = 1
        plot_image(water)

        return water_idx
    dNBR_ds = gdal.Open(f'Datasets/{region}/{region}_dNBR.tif')
    dNBR_img = dNBR_ds.ReadAsArray()
    dNBR = np.array(dNBR_img)

    dNBR_Label = np.zeros_like(dNBR)  # 결과를 저장할 numpy 배열 초기화
    for level in SEVERITY_LEVEL:
        range_min, range_max = SEVERITY_LEVEL[level]['RANGE']
        value = SEVERITY_LEVEL[level]['VALUE']

        if level == 'EnhancedRegrowthHigh':
            dNBR_Label[dNBR < range_max] = value
        elif level == 'HighSeverity':
            dNBR_Label[dNBR >= range_min] = value
        else:
            dNBR_Label[(dNBR >= range_min) & (
                dNBR < range_max)] = value
    
    # Label Water
    if label_water:
        water_idx = find_water(prev)
        dNBR_Label[water_idx] = 7
        im = Image.fromarray(dNBR_Label)
        im.save(f'Datasets/{region}/{region}_dNBR_water_label.tif')
    else:
        im = Image.fromarray(dNBR_Label)
        im.save(f'Datasets/{region}/{region}_dNBR_label.tif')

    return dNBR_Label



def merge_imgs(region,label_water=False):
    _, post = load_images(region)
    if label_water:
        dNBR_Label = load_image(f'Datasets/{region}/{region}_dNBR_water_label.tif')
    else :
        dNBR_Label = load_image(f'Datasets/{region}/{region}_dNBR_label.tif')
        
    dNBR_Label_expand = np.expand_dims(dNBR_Label, axis=0)
    res = np.concatenate((post, dNBR_Label_expand), axis=0)

    if label_water:
        np.save(f'npys/waterLabeled/{region}', res)
    else:   
        np.save(f'npys/{region}', res)

# 50 -> 35 train 15 test


def crop_128(region,label_water=False):
    # [B G R NIR SWIR Target]
    if label_water:
        npy = np.load(f'npys/waterLabeled/{region}.npy')
    else:
        npy = np.load(f'npys/{region}.npy')
    row, col = npy[0].shape
    ir, ic = math.ceil(row/256), math.ceil(col/256)

    res = []
    # 좌상단 시작
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[:, r*128:(r+1)*128, c*128:(c+1)*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 좌 하단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[:, row - (r+1)*128:row - (r)*128, c*128:(c+1)*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 우상단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[:, r*128:(r+1)*128, col-(c+1)*128:col-c*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1
    # 우하단
    count1 = 0
    for r in range(ir):
        for c in range(ic):
            crop = npy[:, row - (r+1)*128:row - (r)*128,
                       col-(c+1)*128:col-c*128]
            res.append(crop)
            # print(count1, np.array(crop).shape)
            count1 += 1

    res = np.array(res)
    if label_water:
        np.save(f'npys/waterLabeled/{region}_{res.shape[0]}', res)
    else:
        np.save(f'npys/{region}_{res.shape[0]}', res)


if __name__ == '__main__':
    # img = load_image('Datasets/Uljin1/Uljin1_dNBR_label.tif')
    # plot_image(img)
    prev, post = load_images(REGION)
    cal_dNBR(prev, post, REGION)
    label_dNBR(REGION,prev,True)
    merge_imgs(REGION,True)
    crop_128(REGION,True)
    '''
    npy = np.load('npys/Donghae_8.npy')
    fig, axs = plt.subplots(4, 2)
    count = 0
    for i in range(4):
        for j in range(2):
            axs[i][j].imshow(npy[count][-1])
            count = count + 1

    plt.show()
    '''
