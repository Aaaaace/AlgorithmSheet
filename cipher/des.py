from typing import Tuple
from typing import List


class DES:
    '''DES加密算法
    外部使用时调用encrypt()和decrypt()即可
    调试时可调用_des_block_encrypt()和_des_block_decrypt()做单个数据块的加解密
    '''

    # 置换矩阵
    IP = [58, 50, 42, 34, 26, 18, 10, 2,
          60, 52, 44, 36, 28, 20, 12, 4,
          62, 54, 46, 38, 30, 22, 14, 6,
          64, 56, 48, 40, 32, 24, 16, 8,
          57, 49, 41, 33, 25, 17, 9, 1,
          59, 51, 43, 35, 27, 19, 11, 3,
          61, 53, 45, 37, 29, 21, 13, 5,
          63, 55, 47, 39, 31, 23, 15, 7]

    FP = [40, 8, 48, 16, 56, 24, 64, 32,
          39, 7, 47, 15, 55, 23, 63, 31,
          38, 6, 46, 14, 54, 22, 62, 30,
          37, 5, 45, 13, 53, 21, 61, 29,
          36, 4, 44, 12, 52, 20, 60, 28,
          35, 3, 43, 11, 51, 19, 59, 27,
          34, 2, 42, 10, 50, 18, 58, 26,
          33, 1, 41, 9, 49, 17, 57, 25]

    E = [32, 1, 2, 3, 4, 5,
         4, 5, 6, 7, 8, 9,
         8, 9, 10, 11, 12, 13,
         12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21,
         20, 21, 22, 23, 24, 25,
         24, 25, 26, 27, 28, 29,
         28, 29, 30, 31, 32, 1]

    PC1 = [57, 49, 41, 33, 25, 17, 9,
           1, 58, 50, 42, 34, 26, 18,
           10, 2, 59, 51, 43, 35, 27,
           19, 11, 3, 60, 52, 44, 36,
           63, 55, 47, 39, 31, 23, 15,
           7, 62, 54, 46, 38, 30, 22,
           14, 6, 61, 53, 45, 37, 29,
           21, 13, 5, 28, 20, 12, 4]

    PC2 = [14, 17, 11, 24, 1, 5,
           3, 28, 15, 6, 21, 10,
           23, 19, 12, 4, 26, 8,
           16, 7, 27, 20, 13, 2,
           41, 52, 31, 37, 47, 55,
           30, 40, 51, 45, 33, 48,
           44, 49, 39, 56, 34, 53,
           46, 42, 50, 36, 29, 32]

    KEY_OFFSET = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    
    S_BOX1 = [14, 0, 4, 15, 13, 7, 1, 4, 2, 14, 15, 2, 11, 13, 8, 1, 3, 10, 10, 6, 6, 12, 12, 11, 5, 9, 9, 5, 0, 3, 7,
              8, 4, 15, 1, 12, 14, 8, 8, 2, 13, 4, 6, 9, 2, 1, 11, 7, 15, 5, 12, 11, 9, 3, 7, 14, 3, 10, 10, 0, 5, 6, 0, 13]
    S_BOX2 = [15, 3, 1, 13, 8, 4, 14, 7, 6, 15, 11, 2, 3, 8, 4, 14, 9, 12, 7, 0, 2, 1, 13, 10, 12, 6, 0, 9, 5, 11, 10,
              5, 0, 13, 14, 8, 7, 10, 11, 1, 10, 3, 4, 15, 13, 4, 1, 2, 5, 11, 8, 6, 12, 7, 6, 12, 9, 0, 3, 5, 2, 14, 15, 9]
    S_BOX3 = [10, 13, 0, 7, 9, 0, 14, 9, 6, 3, 3, 4, 15, 6, 5, 10, 1, 2, 13, 8, 12, 5, 7, 14, 11, 12, 4, 11, 2, 15, 8,
              1, 13, 1, 6, 10, 4, 13, 9, 0, 8, 6, 15, 9, 3, 8, 0, 7, 11, 4, 1, 15, 2, 14, 12, 3, 5, 11, 10, 5, 14, 2, 7, 12]
    S_BOX4 = [7, 13, 13, 8, 14, 11, 3, 5, 0, 6, 6, 15, 9, 0, 10, 3, 1, 4, 2, 7, 8, 2, 5, 12, 11, 1, 12, 10, 4, 14, 15,
              9, 10, 3, 6, 15, 9, 0, 0, 6, 12, 10, 11, 1, 7, 13, 13, 8, 15, 9, 1, 4, 3, 5, 14, 11, 5, 12, 2, 7, 8, 2, 4, 14]
    S_BOX5 = [2, 14, 12, 11, 4, 2, 1, 12, 7, 4, 10, 7, 11, 13, 6, 1, 8, 5, 5, 0, 3, 15, 15, 10, 13, 3, 0, 9, 14, 8, 9,
              6, 4, 11, 2, 8, 1, 12, 11, 7, 10, 1, 13, 14, 7, 2, 8, 13, 15, 6, 9, 15, 12, 0, 5, 9, 6, 10, 3, 4, 0, 5, 14, 3]
    S_BOX6 = [12, 10, 1, 15, 10, 4, 15, 2, 9, 7, 2, 12, 6, 9, 8, 5, 0, 6, 13, 1, 3, 13, 4, 14, 14, 0, 7, 11, 5, 3, 11,
              8, 9, 4, 14, 3, 15, 2, 5, 12, 2, 9, 8, 5, 12, 15, 3, 10, 7, 11, 0, 14, 4, 1, 10, 7, 1, 6, 13, 0, 11, 8, 6, 13]
    S_BOX7 = [4, 13, 11, 0, 2, 11, 14, 7, 15, 4, 0, 9, 8, 1, 13, 10, 3, 14, 12, 3, 9, 5, 7, 12, 5, 2, 10, 15, 6, 8, 1,
              6, 1, 6, 4, 11, 11, 13, 13, 8, 12, 1, 3, 4, 7, 10, 14, 7, 10, 9, 15, 5, 6, 0, 8, 15, 0, 14, 5, 2, 9, 3, 2, 12]
    S_BOX8 = [13, 1, 2, 15, 8, 13, 4, 8, 6, 10, 15, 3, 11, 7, 1, 4, 10, 12, 9, 5, 3, 6, 14, 11, 5, 0, 0, 14, 12, 9, 7,
              2, 7, 2, 11, 1, 4, 14, 1, 7, 9, 4, 12, 10, 14, 8, 2, 13, 0, 15, 6, 12, 10, 9, 13, 0, 15, 3, 3, 5, 5, 6, 8, 11]
    S_BOX = [S_BOX1, S_BOX2, S_BOX3, S_BOX4, S_BOX5, S_BOX6, S_BOX7, S_BOX8]

    P = [16, 7, 20, 21, 29, 12, 28, 17,
         1, 15, 23, 26, 5, 18, 31, 10,
         2, 8, 24, 14, 32, 27, 3, 9,
         19, 13, 30, 6, 22, 11, 4, 25]

    def __init__(self, key: int):
        '''
        Args:
            key: 64位密钥，用十六进制整数表示
        '''
        self.key = key
        self._subkeys = self.__key_schedule()

    def _feistel(self, block: int, round: int):
        '''费斯妥函数

        Args:
            block: 32位数据块，用十六进制整数表示
            round: 轮次，取0~15
        '''

        # 扩张置换(32位->48位)
        block_E = self.__permutate(block, 32, DES.E)

        # 与密钥混合
        block_mixed = block_E ^ self._subkeys[round]

        # S盒(48位->32位)
        block_sbox = 0
        for i in range(8):
            six_bits = block_mixed >> (48 - (i+1) * 6) & 0x3f
            block_sbox <<= 4
            block_sbox += DES.S_BOX[i][six_bits]

        # 置换
        block_P = self.__permutate(block_sbox, 32, DES.P)

        return block_P

    def __key_schedule(self):
        '''密钥调度，生成所有子密钥
        '''

        def rotate_left(key_28bits, bits: int):
            '''循环左移
            密钥调度时使用

            Args:
                key_56bits: 56位密钥
                bits: 移位位数，取1~28
            '''
            return ((key_28bits << bits) | (key_28bits >> 28-bits)) & (0xfffffff)

        subkeys = []

        key_56bits = self.__permutate(self.key, 64, DES.PC1)
        key_left_part = key_56bits >> 28
        key_right_part = key_56bits & 0xfffffff

        for i in range(16):
            key_left_part = rotate_left(key_left_part, self.KEY_OFFSET[i])
            key_right_part = rotate_left(key_right_part, self.KEY_OFFSET[i])

            key_48bits = self.__permutate(
                ((key_left_part << 28) | key_right_part), 56, DES.PC2)
            subkeys.append(key_48bits)

        return subkeys

    def __permutate(self, block: int, bits: int, permutation: List[int]) -> int:
        '''根据输入的置换矩阵对原数据进行置换

        Args:
            block: 需要置换数据块，用十六进制整数表示
            bits: block的位数，block高位为0时，无法判断共有多少位，所以需要标明
            permutation: 置换矩阵
        '''
        block_result = 0
        for i in range(len(permutation)):
            bit = (block >> (bits - permutation[i])) & 1
            block_result <<= 1
            block_result += bit
        return block_result

    def _des_block_encrypt(self, plain_block: int):
        '''单个数据块加密

        Args：
            plain_block： 64位明文数据块，用十六进制整数表示
        '''
        block = self.__permutate(plain_block, 64, DES.IP)

        l = block >> 32
        r = block & 0xffffffff

        for i in range(16):
            l, r = r, l ^ self._feistel(r, i)

        # 最后一轮不做交换
        l, r = r, l
        encrypted_block = l << 32 | r

        return self.__permutate(encrypted_block, 64, DES.FP)

    def _des_block_decrypt(self, encrypted_block: int):
        '''单个数据块解密

        Args:
            encrypted_block: 64位加密数据块，用十六进制整数表示
        '''
        block = self.__permutate(encrypted_block, 64, DES.IP)

        l = block >> 32
        r = block & 0xffffffff

        for i in range(15, -1, -1):
            l, r = r, l ^ self._feistel(r, i)

        # 最后一轮不做交换
        l, r = r, l
        plain_text = l << 32 | r

        return self.__permutate(plain_text, 64, DES.FP)

    def encrypt(self, plain_block: int) -> int:
        '''加密一段明文 TODO
        '''
        return self._des_block_encrypt(plain_block)

    def decrypt(self, encrypted_block: int):
        '''解密一段密文 TODO
        '''
        return self._des_block_decrypt(encrypted_block)


if __name__ == '__main__':
    # 测试数据
    d1 = DES(0x0000000000000000)
    plain_text = 0x0000000000000000
    encypted_text = d1.encrypt(plain_text)
    assert hex(encypted_text) == '0x8ca64de9c1b123a7'
    decypted_text = d1.decrypt(encypted_text)
    assert decypted_text == plain_text

    d2 = DES(0x7CA110454A1A6E57)
    plain_text = 0x01A1D6D039776742
    encypted_text = d2.encrypt(plain_text)
    assert hex(encypted_text) == '0x690f5b0d9a26939b'
    decypted_text = d2.decrypt(encypted_text)
    assert decypted_text == plain_text
