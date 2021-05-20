import numpy as np

def my_erode(bw, mask):
    """
    2018.11.25

    对二值图像的腐蚀
    """
    m, n = bw.shape
    bw_temp = my_convolution2d(bw, mask, 'same')
    threshold = mask.sum()
    for i in range(m):
        for j in range(n):
            if bw_temp[i, j] < threshold:
                bw_temp[i, j] = 0
    return bw_temp.astype(np.bool)