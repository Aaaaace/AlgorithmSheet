#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 该哈希算法是质数取余
# 且关键字只能为正整数
# 初始化时需要参数为一一对应的关键字list和值list
# 构造输入： keys列表，对应的values列表
# 方法：
# 1.     search(self,key):输入整数型关键字key，对应元素频数增一，返回对应的values。
class HashTable(object):

    def prime_generator(self):
        """质数生成器，用于获取一个合适的哈希表长
        """
        is_prime = False
        yield 2
        n = 3
        while True:
            i = 3
            is_prime = False
            while i * i <= n:
                if n % i == 0:
                    is_prime = True
                    break
                i += 2
            if not is_prime:
                yield n
            n += 2

    def __init__(self, keys, values):
        # key与value一一对应
        self._keys = []
        self._values = []
        # 每一个元素的查找时间和查找频率，与key一一对应，用于调试
        self._search_time = []
        self._search_frequency = []
        self._table_length = 0

        # 找到比min_length大的最小质数
        dict_length = len(keys)
        min_length = dict_length * 4 / 3
        g = self.prime_generator()
        for self._table_length in g:
            if self._table_length >= min_length:
                break
        
        # 初始化表（添加_table_length个空元素）
        for i in range(self._table_length):
            self._keys.append(None)
            self._values.append(None)
            self._search_time.append(0)
            self._search_frequency.append(0)
        
        # 放入每个元素
        for key in keys:
            index = key % self._table_length
            while True:
                if self._keys[index] is None:
                    self._keys[index] = key
                    self._values[index] = values[key]
                    self._search_time[index] += 1
                    break
                else:
                    index = (index + 1) % self._table_length
                    self._search_time[index] += 1

    def get(self, key: int):
        if not isinstance(key, int):
            return False
        
        # 查找key所在位置
        index = key % self._table_length
        while self._keys[index] != key and self._keys[index] is not None:
            index = (index + 1) % self._table_length
        if self._keys[index] is not None:
            self._search_frequency += 1
        return self._values[index]


if __name__ == '__main__':
    L1 = {
        123: "Bob",
        214: "Ace",
        125: "Lily",
        489: "Alen",
        741: "Alex",
        419: "Bjh",
        189: "Lwr",
    }  # 测试字典
