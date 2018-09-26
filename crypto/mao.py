import json, base64

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA256


__all__ = [
    'sha256', 'Mao'
]

class Default:
    salt = 'default salt'
    key = 'default key'

def ensure_bytes(data):
    if isinstance(data, str):
        data = data.encode()
    assert isinstance(data, bytes)
    return data



def sha256(data, salt=Default.salt):
    data = ensure_bytes(data)
    salt = ensure_bytes(salt)
    hasher = SHA256.new()
    hasher.update(data + salt)
    return hasher.digest()


class Mao:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls(Default.key)
        return cls.instance
    
    @classmethod
    def encrypt(cls, data):
        return cls.get_instance()._encrypt(data)
    
    @classmethod
    def decrypt(cls, code):
        return cls.get_instance()._decrypt(code)

    def __init__(self, key):
        key = sha256(key)
        self.aes = lambda iv: AES.new(key, AES.MODE_OFB, iv)

    def _encrypt(self, data):
        plain = json.dumps(data).encode()
        padn = 16 - (len(plain) + 16) % 16
        padded = padn * chr(padn).encode() + plain
        iv = Random.new().read(16)
        cipher = self.aes(iv).encrypt(padded)
        code = base64.b64encode(iv + cipher).decode()
        return code

    def _decrypt(self, code):
        iv_cipher = base64.b64decode(code.encode())
        iv = iv_cipher[:16]
        cipher = iv_cipher[16:]
        padded = self.aes(iv).decrypt(cipher)
        plain = padded[padded[0]:]
        data = json.loads(plain)
        return data


if __name__ == '__main__':
    data = 'test data'
    assert data == Mao.decrypt(Mao.encrypt(data))
    data = ['test item', 123]
    assert data == Mao.decrypt(Mao.encrypt(data))
    data = {'test item': 1.123, 3: 'tasdasd'}
    assert data == Mao.decrypt(Mao.encrypt(data))
