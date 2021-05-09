import re


class PasswordPatternCheck(object):
    """密码格式检查
    普通应用(level=1)：
    - 密码长度6位以上


    重要应用(level=2)：
    - 密码长度8位以上
    - 包含字母、数字、特殊符号
    - 不出现连续性字符，如1234, abcd
    - 不连续出现重复字符，如1111, aaaa
    """

    def __init__(self, level=1):
        self.level = level
        self._temp_password = ""

    def _length_check(self) -> bool:
        if len(self._temp_password) == 0 or len(self._temp_password) > 32:
            raise ValueError("密码长度为0或长度大于32")

        if self.level == 1:
            if len(self._temp_password) >= 6:
                return True
            else:
                return False
        elif self.level == 2:
            if len(self._temp_password) >= 8:
                return True
            else:
                return False

    def _character_type_check(self) -> bool:
        password = self._temp_password
        if self.level == 1:
            return True
        elif self.level == 2:
            type_map = {"digit": False, "letter": False, "punctuation": False}
            for c in password:
                c_ascii = ord(c)
                if c_ascii >= 48 and c_ascii <= 57:
                    type_map["digit"] = True
                elif (c_ascii >= 65 and c_ascii <= 90) or (
                    c_ascii >= 97 and c_ascii <= 122
                ):
                    type_map["letter"] = True
                else:
                    type_map["punctuation"] = True
            return all(type_map.values())

    def _continouns_character_check(self) -> bool:
        """检查连续子串长度"""
        password = self._temp_password
        MIN_CONTINOUNS_LENGTH = 3  # 允许连续的子串长度
        if self.level == 1:
            return True
        elif self.level == 2:
            for i in range(len(password)):

                j = i + 1
                continouns_length = 1

                while j <= i + MIN_CONTINOUNS_LENGTH and j < len(password):
                    if ord(password[j - 1]) + 1 == ord(password[j]):
                        continouns_length += 1
                        j += 1
                    else:
                        break

                if continouns_length > MIN_CONTINOUNS_LENGTH:
                    return False

            return True

    def _duplicate_character_check(self) -> bool:
        password = self._temp_password
        MIN_DUPLICATE_LENGTH = 3  # 允许连续的子串长度
        if self.level == 1:
            return True
        elif self.level == 2:
            for i in range(len(password)):

                j = i + 1
                duplicate_length = 1

                while j <= i + MIN_DUPLICATE_LENGTH and j < len(password):
                    if password[j - 1] == password[j]:
                        duplicate_length += 1
                        j += 1
                    else:
                        break

                if duplicate_length > MIN_DUPLICATE_LENGTH:
                    return False

            return True

    def check(self, password: str):
        self._temp_password = password
        if (
            self._character_type_check()
            and self._length_check()
            and self._continouns_character_check()
            and self._duplicate_character_check()
        ):
            return True
        else:
            return False


if __name__ == "__main__":
    ppc = PasswordPatternCheck(level=2)
    pw1 = "1234adfn"
    print(ppc.check(pw1))
    pw2 = "gadgqe(*&lfan"
    print(ppc.check(pw2))
    pw3 = "12gadgqe(*&lfan"
    print(ppc.check(pw3))