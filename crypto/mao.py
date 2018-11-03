import json, base64

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA256


__all__ = ['Mao']

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
    def __init__(self, key=Default.key):
        key = sha256(key)
        self.aes = lambda iv: AES.new(key, AES.MODE_OFB, iv)

    def encrypt(self, data):
        plain = json.dumps(data).encode()
        padn = 16 - (len(plain) + 16) % 16
        padded = padn * chr(padn).encode() + plain
        iv = Random.new().read(16)
        cipher = self.aes(iv).encrypt(padded)
        code = base64.b64encode(iv + cipher).decode()
        return code

    def decrypt(self, code):
        iv_cipher = base64.b64decode(code.encode())
        iv = iv_cipher[:16]
        cipher = iv_cipher[16:]
        padded = self.aes(iv).decrypt(cipher)
        plain = padded[padded[0]:]
        data = json.loads(plain)
        return data


if __name__ == '__main__':

    def test(data):
        mao = Mao()
        cipher = mao.encrypt(data)
        plain = mao.decrypt(cipher)
        print(repr((data, cipher, plain)))
        assert data == plain
    
    for i in range(10):
        test(['s', {'a':1}, 1.2, False] * i)

