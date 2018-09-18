import sys

def binenc(i, zero='0', one='1'):
    i = i.encode() if type(i) is str else i
    assert type(i) is bytes
    s = ''
    for b in i:
        for n in range(8):
            s = (one if int(b) & 1 != 0 else zero) + s
            b >>= 1
    return s

while True:
    data = sys.stdin.buffer.read(64)
    if not data:
        break
    sys.stdout.write(binenc(data, 'a', 'A'))

