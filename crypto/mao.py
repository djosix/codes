import json, base64

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA256


class Mao:
    def __init__(self, secret):
        secret = secret.encode() if type(secret) is str else secret
        sha256 = SHA256.new()
        sha256.update(secret)
        key = sha256.digest()
        self.aes = lambda iv: AES.new(key, AES.MODE_OFB, iv)

    def encrypt(self, data):
        plain = json.dumps(data).encode()
        padn = 16 - (len(plain) + 16) % 16
        padded = padn * chr(padn).encode() + plain
        iv = Random.new().read(16)
        cipher = self.aes(iv).encrypt(padded)
        code = base64.b64encode(iv + cipher)
        return code

    def decrypt(self, code):
        iv_cipher = base64.b64decode(code)
        iv = iv_cipher[:16]
        cipher = iv_cipher[16:]
        padded = self.aes(iv).decrypt(cipher)
        plain = padded[padded[0]:]
        data = json.loads(plain)
        return data


if __name__ == '__main__':
    mao = Mao('secret_key')
    p = {'password': 'secret_password'}
    print('p:', p)
    c = mao.encrypt(p)
    print('c:', c)
    p = mao.decrypt(c)
    print('p:', p)
