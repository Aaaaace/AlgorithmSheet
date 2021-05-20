import numpy as np


def my_dilate(bw, mask):
    """
    <time>2018.11.25</time>

    <summary>对二值图像的膨胀</summary>
    """
    bw_temp = my_convolution2d(bw, mask, 'same')
    return bw_temp.astype(np.bool)