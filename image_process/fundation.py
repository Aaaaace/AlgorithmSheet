#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
'''
author: storyoftime
email: wsk_8004@qq.com
'''
import logging
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片


# 数字图像处理
## 应用方法
def my_threshold_segmentation(bw, threshold):
    '''
    <time>2018.12.2</time>
    <summary>阈值分割，阈值可输入相对值(百分比)也可输入绝对值</summary>
    <param name="bw">输入图像</param>
    <param name="threshold">阈值，若为不超过1的小数，则视为相对阈值，若为大于1的数，则视为觉得阈值</param>
    '''
    if threshold < 0:
        raise ValueError('Threshold must be a positive num')
    elif threshold <= 1:
        max_value = max(bw.flat)
        min_value = min(bw.flat)
        threshold = (max_value-min_value)*float(threshold) + float(min_value)

    m ,n = bw.shape
    bw0 = bw.copy()
    bw0.flags.writeable = True
    
    for i in range(m):
        for j in range(n):
            if bw0[i, j] < threshold:
                bw0[i, j] = 0
    return bw0.astype(np.bool)
def my_edge(bw, threshold):
    """
    2018.11.26
    
    检测图像边缘
    返回检测到的边缘二值图像
    阈值用于消去检测到的噪声
    
    时间复杂度：
    
    Args：
        bw: a grey-scale image with 8-bit depth
        threshold: a decimal between 0 and 1
    Returns:
        bw_edge_binary: a binary image with the detected edge
    Raises:

    """
    m, n = bw.shape
    bw0 = bw.astype(np.int16)
    bw_edge_rows = np.zeros([m, n])
    bw_edge_cols = np.zeros([m, n])
    for i in range(m-1):
        bw_edge_rows[i, :] = abs(bw0[i+1, :] - bw0[i, :])
    bw_edge_rows[m-1, :] = 0
    for j in range(n-1):
        bw_edge_cols[:, j] = abs(bw0[:, j+1] - bw0[:, j])
    bw_edge_cols[:, n-1] = 0

    bw_edge = np.sqrt(bw_edge_cols*bw_edge_cols + bw_edge_rows*bw_edge_rows)
    index_threshold = bw_edge.max()*threshold
    bw_edge_binary = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            if bw_edge[i, j] > index_threshold:
                bw_edge_binary[i, j] = 1
                
    return bw_edge_binary
def my_edge_deprecated(bw, threshold):
    """
    2018.11.26
    
    检测图像边缘
    返回检测到的边缘二值图像
    阈值用于消去检测到的噪声
    
    时间复杂度：
    
    Args：
        bw: a grey-scale image with 8-bit depth
        threshold: a decimal between 0 and 1
    Returns:
        bw_edge_binary: a binary image with the detected edge
    Raises:

    """
    m, n = bw.shape
    bw0 = bw.astype(np.float16)
    bw_edge_rows = abs(bw0[1:m, :] - bw0[0:m-1, :])
    bw_edge_rows = np.vstack((bw_edge_rows, np.zeros(n, dtype=np.float16)))
    
    bw_edge_cols = abs(bw0[:, 1:n] - bw0[:, 0:n-1])
    bw_edge_cols = np.hstack((bw_edge_cols, np.zeros((m, 1))))
    
    bw_edge = np.sqrt(bw_edge_cols*bw_edge_cols + bw_edge_rows*bw_edge_rows)
    index_threshold = bw_edge.max()*threshold
    bw_edge_binary = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            if bw_edge[i, j] > index_threshold:
                bw_edge_binary[i, j] = 1
                
    return bw_edge_binary
def my_single_circle_detect_least_squares(bw):
    """
    2018.11.25
    
    检测单个圆，返回圆心坐标、半径

    Args：
        bw: a binary image containing edge information
    Returns:
        a matrix of circle's radius and center coordinata which format
        like [r, a, b]
    Raises:

    """
    try:
        m, n = bw.shape
    except ValueError:
        raise ValueError("the input array(map) expects to be "
                         "a binary image")
    edge_pixels = []  # 边缘点的坐标
    vb = []
    va = []
    for i in range(m):
        for j in range(n):
            if bw[i, j] > 0:
                edge_pixels.append([i, j])
    for pixel in edge_pixels:
        vb.append([pixel[0]*pixel[0]+pixel[1]*pixel[1]])
        va.append([1, 2*pixel[0], 2*pixel[1]])
    vb = np.mat(vb)
    va = np.mat(va)
    vx = va.I * vb
    vx[0] = np.sqrt(vx[0] + vx[1]*vx[1] + vx[2]*vx[2])
    return np.asarray(vx)
def my_single_circle_detect_symmetry(bw):
    '''
    2018.11.25
    
    '''
    try:
        m, n = bw.shape
    except ValueError:
        raise ValueError("the input array(map) expects to be "
                         "a binary image")
    edgeNum = 0  # 边缘点的数量
    for i in range(m):
        for j in range(n):
            if bw[i, j] > 0:
                edgeNum += 1
    edgeCoordinate = np.zeros([edgeNum, 2],dtype=np.uint32)  # 边缘点的坐标
    edgeIndex = 0  # 边缘上的点序号
    for i in range(m):
        for j in range(n):
            if bw[i, j] > 0:
                edgeCoordinate[edgeIndex] = [i, j]
                edgeIndex += 1
    a = np.sum(edgeCoordinate[:, 0])/edgeNum
    b = np.sum(edgeCoordinate[:, 1])/edgeNum
    r = 0.0
    for pixel in edgeCoordinate:
        deltax=pixel[0]-a
        deltay=pixel[1]-b
        r += np.sqrt(deltax * deltax + deltay * deltay)
    r /= edgeNum
    return np.array([[r], [a], [b]])
def my_gamma(bw, gamma):
    """
    2018.11.25
    gamma校正
    Args：
        bw: a 8-bit grey-scale image containing edge information
        gamma: gamma value, positive
    Returns:
        The processed grey-scale map
    Raises:
    """
    bw0 = bw.astype(np.double)
    return (((bw0/255) ** gamma) * 255).astype(np.uint8)
def my_hist_equalization(bw):
    """
    2018.11.25
    
    histogram equalization(only for grey-scale maps)

    :param bw: a 8-bit grey-scale image
    :return: image after equalled, histogram before, histogram now
    """
    mean = bw.size/255.0
    try:
        m, n = bw.shape
    except ValueError:
        raise ValueError("the input array(map) expects to be "
                         "a 8-bit grey-scale image")

    histbefore = np.zeros(256, dtype=np.uint32)
    histnow = np.zeros(256, dtype=np.uint32)
    # 原图与均衡之后颜色值的对应关系
    correspondence = np.zeros(256, dtype=np.uint32)
    for pixel in bw.flat:
        histbefore[pixel] += 1
    # correspondence
    cumulative_distribution = 0
    for index, val in enumerate(histbefore):
        cumulative_distribution += val
        index_now = np.floor(cumulative_distribution/mean)
        correspondence[index] = index_now

    bw_equalization = bw.copy()
    for i in range(m):
        for j in range(n):
            temp = correspondence[bw[i, j]]
            bw_equalization[i, j] = temp
            histnow[temp] += 1
    return bw_equalization, histbefore, histnow
def my_hist_stretching(bw):
    """
    2018.11.25
    
    stretch the color histogram to the whole scale

    :param bw: a 8-bit grey-scale map
    :return: the processed image
    """
    index_max = bw.max()
    index_min = bw.min()
    scale = index_max - index_min
    bw0 = bw.astype(np.double)
    bw0 = (((bw0 - index_min)/scale)*255).astype(np.uint8)
    return bw0
def my_bit_plane_separator(bw):
    """
    未测试
    <time>2018.11.25</time>
    <summary>separate a image into the bit planes</summary>
    :param bw: a 8-bit grey-scale map
    :return: a list, containing 8 bit plane maps
    """
    bit_plane = []
    bw0 = bw.copy()
    for _ in range(8):
        bit_plane.append(bw0.mod(2))
        bw0 = bw0 / 2
    return bit_plane
def my_sample_deprecated(bw, k_m, k_n, mode='single'):
    """
    2018.11.25
    
    图像取样，减小分辨率
    :param bw: 输入图像
    :param k_m, k_n: 取样前后的行、列数比，为整数
    :param mode:
    option: 'single', sample a single point
            'average', sample the averge of the area
    """
    m, n = bw.shape
    m_sample = int(np.ceil(m/k_m))
    n_sample = int(np.ceil(n/k_n))
    bw_sample = np.zeros((m_sample, n_sample), dtype=np.uint8)
    if mode == 'single':
        for i in range(m_sample):
            for j in range(n_sample):
                bw_sample[i, j] = bw[i*k_m, j*k_n] 
    elif mode == 'average':
        pointnum = k_m*k_n
        for i in range(m_sample):
            for j in range(n_sample):
                lbm = i*k_m
                ubm = lbm + k_m
                lbn = j*k_n
                ubn = lbn + k_n
                bw_sample[i, j] = np.rint(bw[lbm: ubm, lbn: ubn].sum()/pointnum)
    else:
        raise TypeError('mode should be ')
    return bw_sample
def my_sample(bw, k_m, k_n, mode='single'):
    """
    未测试
    2018.11.25
    
    图像取样，减小分辨率
    :param bw: 输入图像
    :param k_m, k_n: 取样前后的行、列数比，为整数
    :param mode:
    option: 'single', sample a single point
            'average', sample the averge of the area
    """
    m, n = bw.shape
    m_sample = int(np.ceil(m/k_m))
    n_sample = int(np.ceil(n/k_n))
    bw_sample = np.zeros((m_sample, n_sample), dtype=np.uint8)
    if mode == 'single':
        bw_sample = bw[::k_m, ::k_n]
    elif mode == 'average':
        pointnum = k_m*k_n
        for i in range(m_sample):
            for j in range(n_sample):
                lbm = i*k_m
                ubm = lbm + k_m
                lbn = j*k_n
                ubn = lbn + k_n
                bw_sample[i, j] = np.rint(bw[lbm: ubm, lbn: ubn].sum()/pointnum)
    else:
        raise TypeError('mode should be ')
    return bw_sample
def my_interpolate(bw, k_m, k_n, mode='bilinear'):
    """
    2018.11.26
    遗留问题：双线性插值图像右边和下边会有一条颜色与原图像最下行像素相同的色带
    
    图像插值，增大分辨率
    :param bw: 输入图像
    :param k_m, k_n: 取样后前的行、列数比，为整数
    :param mode:
    option: 'bilinear', bilinear interpolation
            'single', single point value interpolation
    """
    m, n = bw.shape
    bw0 = bw.astype(np.float32)
    bw_interpolation = np.zeros((m*k_m, n*k_n), dtype=np.uint16)
    if mode == 'single':
        for i in range(m):
            for j in range(n):
                bw_interpolation[i*k_m:i*k_m+k_m, j*k_n:j*k_n+k_n] = bw[i,j]
    elif mode == 'bilinear':
        # 行间线性插值
        delta = ((bw0[1:m,:] - bw0[0:m-1,:])/k_m)
        delta = np.vstack((delta, np.zeros(n, dtype=np.float32)))
        for i in range(0,k_m):
            bw_interpolation[i::k_m,::k_n] = delta*i + bw0
        # 列间线性插值
        start = bw_interpolation[:,::k_n].astype(np.float32)
        delta = ((start[:,1:] - start[:,:n-1])/k_n)
        delta = np.hstack((delta, np.zeros((m*k_m,1),dtype=np.float32)))
        for j in range(1,k_n):
            bw_interpolation[:,j::k_n]=delta*j+start
    return bw_interpolation
## 滤波器
def my_mean_filter(bw, window):
    '''
    未测试
    <time>2018.11.29</time>
    <summary>均值滤波器，可加权</summary>
    <param name="bw">输入图像</param>
    <param name="window">滤波窗口及滤波窗口中每个位置的权值</param>
    '''
    weight = window.sum()
    
    bw_filtered = \
        (my_convolution2d(bw, window, mode='custom')/weight).astype(np.uint8)
    return bw_filtered
def my_median_filter(bw, window):
    '''
    未测试
    未完成
    <time>2018.11.29</time>
    <summary>中值滤波器</summary>
    <param name="bw">输入图像</param>
    <param name="window">滤波窗口（只取大小）</param>
    '''
    pass    
## 形态学方法
def my_erode(bw, mask):
    """
    <time>2018.11.25</time>
    
    <summary>对二值图像的腐蚀</summary>
    """
    m, n = bw.shape
    bw_temp = my_convolution2d(bw, mask, 'same')
    threshold = mask.sum()
    for i in range(m):
        for j in range(n):
            if bw_temp[i,j]<threshold:
                bw_temp[i,j] = 0
    return bw_temp.astype(np.bool)
def my_dilate(bw, mask):
    """
    <time>2018.11.25</time>
    
    <summary>对二值图像的膨胀</summary>
    """
    bw_temp = my_convolution2d(bw, mask, 'same')
    return bw_temp.astype(np.bool)
def my_bwlabel(bw, mode='four'):
    '''
    未完成
    <time>2018.11.27</time>
    <summary>二值图像标记连通域</summary>
    '''
    class Runsofline(object):
        '''
        <summary>保存每行中的团(run)的信息</summary>
        '''
        flag = 1
        def __init__(self, lineNo):
            '''
            <summary>构造函数</summary>
            <param name="lineNo">行号</param>
            '''
            # 行号，表示是第几行的runset
            self.lineNo = lineNo
            # flag of each run
            self.run_flags = []
            # start of each run
            self.run_starts = []
            # end of each run
            self.run_ends = []
            # number of run
            self.run_number = 0
        
        @classmethod
        def scanline(cls, lineNo, linedata, linelen):
            '''
                <summary>扫描一行</summary>
                <param name="lineNo">行号</param>
                <param name="linedata">该行数据，为一维ndarray</param>
                <param name="linelen">该行长度</param>
            '''
            runs = Runsofline(lineNo)
            if linedata[0]:
                runs.run_starts.append(0)
                runs.run_flags.append(cls.flag)
                
            for i in range(1, linelen):
                if linedata[i]:
                    if linedata[i-1]:
                        continue
                    else:
                        runs.run_starts.append(i)
                        runs.run_flags.append(cls.flag)
                else:
                    if linedata[i-1]:
                        runs.run_ends.append(i)
                        cls.flag += 1
                        runs.run_number += 1
            return runs

    m, n = bw.shape
    # 得到每一行的runs信息
    runset = []
    for i in range(m):
        runset.append(Runsofline.scanline(i, bw[i], n))
    
    # 分析runset
    
    
    pass
## 常用过程
def my_nearest_filling(bw, window):
    '''
    未测试
    <time>2018.12.1</time>
    <summary>对图像进行最近像素填充，返回填充好的图像</summary>
    <param name="bw">输入图像</param>
    <param name="window">输入窗口（只取大小）</param>
    
    <example>
        bw = np.array([[1,2,3,4], \
                       [5,6,7,8]],dtype=np.uint8)
        window = np.array([[0, 0],
                           [0 ,0]])
        
        return: 
    </example>
    '''
    m ,n = bw.shape
    m_window, n_window = window.shape
    m_Y, n_Y = (m + m_window - 1, n + n_window - 1)
    m_map, n_map = (m + 2*m_window - 2, n + 2*n_window - 2)
    map = np.zeros((m_map, n_map))
    map[m_window-1:m_Y, n_window-1:n_Y] = bw
    
    # 边界填充
    for i in range(m_window-1):
        map[i,n_window-1:n_Y] = bw[0]
    for i in range(m_Y+1,m_map):
        map[i,n_window-1:n_Y] = bw[-1]
    for j in range(n_window-1):
        map[m_window-1:m_Y,j] = bw[:,0]
    for j in range(n_Y+1,n_map):
        map[m_window-1:m_Y,j] = bw[:,-1]
    map[:m_window-1, :n_window-1] = bw[0, 0]
    map[:m_window-1, -(n_window-1):] = bw[0, -1]
    map[-(m_window-1):, :n_window-1] = bw[-1, 0]
    map[-(m_window-1):, -(n_window-1):] = bw[-1, -1]
    
    return map

# 高数
## 基础运算(卷积)
def my_convolution1d(A, B):
    '''
    <time>2018.11.25</time>
    <summary>Y[n] = A[n]*B[n] 一维卷积
    ps: 使用概念实现
    时间复杂度：O(n_A*n_B)</summary>
    '''
    lenA = len(A)
    lenB = len(B)
    
    if lenA < lenB:
        temp = B
        B = A
        A = temp
        lenA = len(A)
        lenB = len(B)
    lenY = lenA + lenB - 1
    Y = np.zeros(lenY)
    # map = [(x, y) for x in range(lenA) for y in range(lenB)]
    # for循环中A长度必须大于等于B
    for n in range(lenY):
        ubA = min(n+1, lenA)  # 重叠部分下标上限（保证下标不超过A、B的长度）
        lbB = max(0, n+1-lenB)  # 重叠部分下标下限（保证下标不为负）
        for i in range(lbB, ubA):
            Y[n] += A[i]*B[n-i]
    return Y 
def my_convolution1d_deprecated(A, B):
    '''
    <time>2018.11.25</time>
    <summary>Y[n] = A[n]*B[n] 卷积
    时间复杂度：O(n_A*n_B)
    简单方法实现
    ps: 但是超久，弃之</summary>
    '''
    lenA = len(A)
    lenB = len(B)
    
    if lenA < lenB:
        temp = B
        B = A
        A = temp
        lenA = len(A)
        lenB = len(B)
    lenY = lenA + lenB - 1
    Y = np.zeros(lenY)
    
    for n in range(lenY):
        for i in range(n+1):
            try:
                Y[n] += A[i]*B[n-i]
            except IndexError:
                continue
    return Y  
def my_convolution2d_deprecated(A, B, mode='full', *arg):
    '''
    <time>2018.11.25</time>
    <summary>Y[m, n] = A[m, n]*B[m, n]二维卷积
             由定义公式实现</summary>
    '''
    if mode != 'full' and mode != 'same':
        raise TypeError(r'mode should be "full" or "same".')
    
    m_A, n_A = A.shape
    m_B, n_B = B.shape
    m_Y, n_Y = (m_A + m_B - 1, n_A + n_B - 1)
    
    Y = np.zeros((m_Y, n_Y))
    
    # count = 0  # 测试用的计数变量
    for m in range(m_Y):
        ubm = min(m+1, m_A)
        lbm = max(0, m+1-m_B)
        for n in range(n_Y):
            ubn = min(n+1, n_A)
            lbn = max(0, n+1-n_B)
            for i in range(lbm, ubm):
                for j in range(lbn, ubn):
                    Y[m, n] += A[i, j]*B[m-i, n-j]
    if mode == 'full':
        return Y
    elif mode == 'same':
        xlb = int(np.floor((m_B+1)/2))
        xub = xlb + m_A
        ylb = int(np.floor((n_B+1)/2))
        yub = ylb + n_A
        return Y[xlb:xub, ylb:yub]
def my_convolution2d(A, B, mode='full'):
    '''
    <time>2018.12.1</time>
    <summary>Y[m, n] = A[m, n]*B[m, n]二维卷积
             窗口移动（图形）思路实现
    </summary>
    <param name="mode">卷积结果模式：
                       'full':返回完整的卷积结果，通常适用于数学运算
                       'same':返回与A大小相同的卷积结果（通过截取），通常用于图像处理
                       'custom':可通过arg传递其他参数，用于某些函数的实现
    </param>
    <param name="arg">传递其他的参数，用于某些函数的实现</param>
    '''
    if mode != 'full' and mode != 'same':
        raise TypeError(r'mode should be "full" or "same".')
    
    m_A, n_A = A.shape
    m_B, n_B = B.shape
    m_Y, n_Y = (m_A + m_B - 1, n_A + n_B - 1)
    m_map, n_map = (m_A + 2*m_B - 2, n_A + 2*n_B - 2)
    map = np.zeros((m_map, n_map))
    map[m_B-1:m_Y, n_B-1:n_Y] = A
    Y = np.zeros((m_Y, n_Y))
    
    # 翻转B
    B = B[::-1,::-1]
    
    for i,j in [(i,j) for i in range(m_Y) for j in range(n_Y)]:
        Y[i, j] = (B*map[i:i+m_B, j:j+n_B]).sum()

    if mode == 'full':
        return Y
    elif mode == 'same':
        xlb = int(np.floor((m_B+1)/2))
        xub = xlb + m_A
        ylb = int(np.floor((n_B+1)/2))
        yub = ylb + n_A
        return Y[xlb:xub, ylb:yub]

# 线性代数
# 基础运算(线性代数)
def my_matrix_determinant(A):
    '''
    求n阶矩阵的行列式
    实现思路: 代数余子式的方法
    Args:
        A: 输入的n阶矩阵
    Returns:
        det: 行列式的值
    Raise:
        ValueError
    '''
    m_A, n_A = A.shape
    if m_A and n_A and m_A != n_A:
        raise ValueError('输入必须为一个n阶矩阵')
    if m_A == 1:
        return A[0, 0]
    elif m_A == 2:
        return A[0,0]*A[1,1] - A[0,1]-A[1,0]
    elif m_A == 3:
        det = A[0,0]*A[1,1]*A[2,2] + A[0,2]*A[1,0]*A[2,1] + A[0,1]*A[1,2]*A[2,0]
        det -= A[0,2]*A[1,1]*A[2,0] + A[0,0]*A[1,2]*A[2,1] + A[0,1]*A[1,0]*A[2,2]
        return det
    else:
        det = 0
        for i in range(m_A):
            M = np.vstack((A[0:i, 1:], A[i+1:, 1:]))
            det += ((-1)**i) * A[i, 0] * my_matrix_determinant(M)
        return det
def my_matrix_transpose(A):
    '''
    未测试
    矩阵转置 Matrix transpose
    (只能输入二维list)
    Args:
        A: 输入矩阵
    Returns:
        AT: A的转置矩阵
    '''
    if isinstance(A, Iterable):
        if isinstance(A[0], Iterable):
            if not isinstance(A[0][0], Iterable):
                return np.matrix(list(zip(*A)))
    print('该函数现在只能接收一个二维列表作为输入矩阵')
    raise ValueError('Function my_transpose expects a matrix')
    
def my_matrix_inverse(A):
    '''
    矩阵求逆
    实现思路: 求出矩阵的伴随矩阵在除以矩阵的行列式
    Args:
        A: 输入矩阵
    Returns:
        Ai: 输入矩阵的逆矩阵
    '''
    try:
        det = my_matrix_determinant(A)
        if det == 0:
            return None
    except:
        raise
    m_A, n_A = A.shape
    Ai = np.zeros((m_A, n_A)).astype(np.float)
    for i in range(m_A):
        for j in range(n_A):
            M12 = np.hstack((A[0:i, 0:j], A[0:i, j+1:]))
            M34 = np.hstack((A[i+1:, 0:j], A[i+1:, j+1:]))
            M = np.vstack((M12, M34))
            Ai[j, i] = ((-1)**(i+j))*my_matrix_determinant(M)
    return Ai/det
def my_matrix_rank(A):
    '''
    未实现
    求矩阵的秩
    实现思路:
    Args:
        A: 输入矩阵
    Returns:
        rank: 矩阵的秩
    '''
    A_0 = A.copy()
    m_A, n_A = A.shape
    pass
