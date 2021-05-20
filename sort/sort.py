#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def insert_sort(inputlist):
    """
    直接插入排序，升序

    :param inputlist: a list of number
    :return: the ascending list
    """
    length = len(inputlist)
    for x in range(1, length):
        key = inputlist[x]  # 要插入的数
        i = x - 1
        while i >= 0:
            if inputlist[i] > key:
                inputlist[i + 1]=inputlist[i]
            else:
                break
            i -= 1
        inputlist[i + 1] = key
    return inputlist


def bubble_sort(inputlist):
    """
    冒泡排序

    :param inputlist: a list of number
    :return: the ascending list
    """
    length = len(inputlist)
    for j in range(1, length):
        for i in range(length-j):
            if inputlist[i] > inputlist[i+1]:
                inputlist[i], inputlist[i+1] = inputlist[i+1], inputlist[i]
    return inputlist


def select_sort(inputlist):
    """
    简单选择排序

    :param inputlist: a list of number
    :return: the ascending list
    """
    length = len(inputlist)
    for i in range(length):
        minimum = i
        for j in range(i, length):
            if inputlist[j] < inputlist[minimum]:
                minimum = j
        inputlist[i], inputlist[minimum] = inputlist[minimum], inputlist[i]
    return inputlist


def shell_sort(inputlist):
    pass
    return inputlist


def quick_sort(inputlist):
    pass
    return inputlist


# 堆排序
def heap_sort(inputlist):
    def heap_rebuild(inputlist, index, length):
        """
        堆排序

        :param inputlist:
        :param index:
        :param length:
        :return:
        """
        lchild = index * 2 + 1
        rchild = lchild + 1
        if lchild < length:
            heap_rebuild(inputlist, lchild, length)
            if inputlist[lchild] > inputlist[index]:
                inputlist[index], inputlist[lchild] = inputlist[lchild], inputlist[index]
        if rchild < length:
            heap_rebuild(inputlist, rchild, length)
            if inputlist[rchild] > inputlist[index]:
                inputlist[index], inputlist[rchild] = inputlist[rchild], inputlist[index]

    length = len(inputlist)
    # 重建完之后最大的元素在下标为0的位置
    heap_rebuild(inputlist, 0, length)
    while length > 0:
        # inputlist[length-1]为末尾元素
        length -= 1
        # 交换到表尾
        inputlist[0], inputlist[length] = inputlist[length], inputlist[0]
        heap_rebuild(inputlist, 0, length)
    return inputlist


L1 = [5, 9, 6]
heap_sort(L1)
print(L1)
