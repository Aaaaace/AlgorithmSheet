#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np


# 基础算法
def C(m, n):
    '''
    <time>2018.11.27</time>
    <summary>组合</summary>
    '''
    result = 1
    if m*n > 0 and isinstance(m*n, int) and n > m:
        for i in range(n-m+1, n+1):
            result *= i
        for i in range(1, m+1):
            result /= i
    return result
    
def A(m, n):
    '''
    <time>2018.11.27</time>
    <summary>排列</summary>
    '''
    result = 1
    if m*n > 0 and isinstance(m*n, int) and n > m:
        for i in range(n-m+1, n+1):
            result *= i
    return result
    
# 规划算法
def my_simplex_method(f, A, b, Aeq, beq, lb, ub):
    '''
    未完成
    <time>2018.11.27</time>
    <summary>单纯型法，用于线性规划</summary>
    
    '''
    pass
    