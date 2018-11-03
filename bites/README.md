# bites

This package provides a useful class: `Bs` (Bytes), to help you do several operations on bytes without writing a lot of stupid codes.


- [bites](#bites)
    - [Install](#install)
    - [Why?](#why)
        - [When you want to conduct a simple xor](#when-you-want-to-conduct-a-simple-xor)
        - [When you need a simple encryption](#when-you-need-a-simple-encryption)
    - [Usage](#usage)
        - [Creating Bs](#creating-bs)
            - [Using the constructor](#using-the-constructor)
            - [From integer](#from-integer)
            - [From hex string](#from-hex-string)
            - [From bit string](#from-bit-string)
            - [From file](#from-file)
            - [From base64 encoded bytes](#from-base64-encoded-bytes)
            - [Random](#random)
        - [Basic Operations](#basic-operations)
            - [Slicing](#slicing)
            - [Setting values for slice of Bs](#setting-values-for-slice-of-bs)
            - [Other useful methods](#other-useful-methods)
        - [Bytewise Operations](#bytewise-operations)
            - [Eamples](#eamples)
            - [Binary operations with other types](#binary-operations-with-other-types)
            - [Other useful methods](#other-useful-methods)
        - [Bitwise Operations](#bitwise-operations)
            - [Basic properties](#basic-properties)
            - [Logical operations](#logical-operations)
            - [Shifting over all bits](#shifting-over-all-bits)
            - [Other useful methods](#other-useful-methods)
        - [Convertions](#convertions)
            - [To integer](#to-integer)
            - [To string](#to-string)
        - [Other](#other)
            - [Base64 encode](#base64-encode)
            - [Hashing](#hashing)


## Install

```shell
python3 -m pip install -U bites
```

## Why?

Aren't you mad?

#### When you want to conduct a simple xor

```python
a = b'deadbeef'
b = b'faceb00k'

# Standard
c = bytes(i ^ j for i, j in zip(a, b))

# NumPy
import numpy as np
c = (np.array([*a]) ^ np.array([*b])).astype(np.uint8).tobytes()

# Using this package
from bites import Bs
c = bytes(Bs(a) ^ b)
```

#### When you need a simple encryption

```python
m = 'the_plain_text_you_wanna_encrypt'
e = 'the_secret_key'

# Standard
mb = m.encode()
eb = e.encode()
mbl = len(mb)
ebl = len(eb)
l = max(mbl, ebl)
cb = bytes([mb[i % mbl] ^ eb[i % ebl] for i in range(l)])

# NumPy
import numpy as np
mb = np.array(list(m.encode()))
eb = np.array(list(e.encode()))
print('You should repeat these arrays to the same length to use xor, '
      "and you gived up just because you don't know which method to use.\n"
      'You start googling then write down this code:')
eb_ = np.tile(eb, mb.size // eb.size + 1)[:mb.size]
cb = (mb ^ eb_).astype(np.uint8).tobytes() # elegant!

# After you found this package:
from bites import Bs
cb = bytes(Bs(m) ^ e)

# Or, if you don't want auto-repeating / padding, use n()
cb = bytes(Bs(m) ^ Bs(e).n())   # error!!!
cb = bytes(Bs(m) ^ Bs(e).r())   # repeat e to fit the length of m
cb = bytes(Bs(m) ^ Bs(e).p(0))  # pad e with 0s to fit the length of m
```

- `Bs` by default uses auto-repeating.
- `bs.r()` explicitly specifies using auto-repeating.
- `bs.p(c)` returns a `Bs` that pads `c` to fit the other longer operands by default.
- `bs.n()` returns a `Bs` object that will not change its length automatically.

## Usage

```python
from bites import Bs

bs = Bs(1, 2, 3, 4)
bs # <Bs 4: [01, 02, 03, 04]>
```

### Creating Bs

#### Using the constructor

These lines create the same thing, the input parameters will be flattened and automatically converted to ints in `range(0, 256)`.

```python
# These all create `<Bs 4: [01, 02, 03, 04]>`
Bs(1, 2, 3, 4)
Bs([1, 2, 3, 4])
Bs([[1], [2], [3, [4]]])
Bs(256+1, 256+2, 256+3, 256+4)
Bs(1-256, 2-256, 3-256, 4-256)
Bs(bytes([1, 2]), 3, 4)
Bs('\x01', b'\x02', 3, [4])
Bs(Bs(1, 2), [3], 4)
Bs(Bs(1, Bs(2)), Bs([3, Bs(4)]))
```

Simple rules

- `int` will be replaced with its remainder of 256.
- `str` will be encoded into `bytes` (UTF-8).
- `Iterable` will be flattened.

```python
>>> Bs(range(5))
<Bs 5: [00, 01, 02, 03, 04]>

>>> Bs([i for i in range(256) if i % 3 == i % 7 == 0 ])
<Bs 13: [00, 15, 2a, 3f, 54, 69, 7e, 93, a8, bd, d2, e7, fc]>

>>> Bs(map(lambda n: n + 3, range(5)))
<Bs 5: [03, 04, 05, 06, 07]>

>>> Bs(range(0, 3), range(10, 13))
<Bs 6: [00, 01, 02, 0a, 0b, 0c]>
```

#### From integer

```python
# Integers will be considered as little endien
>>> Bs.from_int(8192)
<Bs 2: [00, 20]>
>>> Bs.from_int(0x102030)
<Bs 3: [30, 20, 10]>

# Simply call `bs.rev()` if you want big endian
>>> Bs.from_int(8192).rev()
<Bs 2: [20, 00]>
```

#### From hex string

```python
# 'DE' is the first byte
>>> Bs.from_hex('DEADBEEF')
<Bs 4: [de, ad, be, ef]>

# If the string starts with '0x', 'EF' will be the first byte
>>> Bs.from_hex('0xDEADBEEF')
<Bs 4: [ef, be, ad, de]>
>>> Bs.from_int(int('0xDEADBEEF', 16))
<Bs 4: [ef, be, ad, de]>
```

#### From bit string

```python
# The first bit is LSB
>>> Bs.from_bin('00001111')
<Bs 1: [f0]>

# If the string starts with '0b', the first bit in the string is MSB
>>> Bs.from_bin('0b00001111')
<Bs 1: [0f]>
>>> Bs.from_int(int('0b00001111', 2))
<Bs 1: [0f]>

# Notice that this will not be '00001111'
>>> Bs.from_bin('0b00001111').bin()
'11110000'
```

#### From file

```python
print(Bs.load('/etc/passwd').str())
```

#### From base64 encoded bytes

```python
Bs.from_base64('ZnVjaw==')
```

#### Random

```python
>>> import string

>>> Bs.rand(8, cs=(string.ascii_lowercase + '0123456')).str()
'c4u0epdn'

>>> Bs.rand(8, cs=range(100)).hex()
'4a334519435d1103'

>>> Bs.rand(8, cs=string.hexdigits).str()
'cb41fA41'
```

### Basic Operations

#### Slicing

```python
>>> bs = Bs(1, 2, 3, 4)
>>> bs
<Bs 4: [01, 02, 03, 04]>

>>> bs[:2]
<Bs 2: [01, 02]>

>>> bs[2]
<Bs 1: [03]>

>>> bs[-1]
<Bs 1: [04]>

>>> bs[::-1]
<Bs 4: [04, 03, 02, 01]>

>>> bs[::2]
<Bs 2: [01, 03]>
```

#### Setting values for slice of Bs

```python
>>> bs = Bs(1, 2, 3, 4)
>>> bs
<Bs 4: [01, 02, 03, 04]>

>>> bs[:2] = 0
>>> bs
<Bs 4: [00, 00, 03, 04]>

>>> bs[:] = 0
>>> bs
<Bs 4: [00, 00, 00, 00]>

>>> bs[:] = '1234'
>>> bs
<Bs 4: [31, 32, 33, 34]>

>>> bs[:] = '123'
>>> bs
<Bs 4: [31, 32, 33, 31]>

>>> bs[:] = '12345'
>>> bs
<Bs 4: [31, 32, 33, 34]>

>>> bs[:] = Bs('12').n()
# Error: cannot set values to range(0, 4): r=False, p=None, l=2

>>> bs[:] = Bs('12').p(0)

>>> bs
<Bs 4: [31, 32, 00, 00]>
```

#### Other useful methods

```python
>>> bs = Bs('dead')
>>> bs
<Bs 4: [64, 65, 61, 64]>

# Repeat n times
>>> bs.rep(2)
<Bs 8: [64, 65, 61, 64, 64, 65, 61, 64]>

# Repeat to length
>>> bs.repto(6)
<Bs 6: [64, 65, 61, 64, 64, 65]>

# Pad to length
>>> bs.padto(6, 0)
<Bs 6: [64, 65, 61, 64, 00, 00]>

# Append or concatenate
>>> bs @ 'beef'
<Bs 8: [64, 65, 61, 64, 62, 65, 65, 66]>

# Extend to length automatically
>>> bs.extto(6)
<Bs 6: [64, 65, 61, 64, 64, 65]>

# Explicit automatic repeating
>>> bs.r().extto(6)
<Bs 6: [64, 65, 61, 64, 64, 65]>

# Use automatic padding
>>> bs.p(0).extto(6)
<Bs 6: [64, 65, 61, 64, 00, 00]>

# Disable automatic extension
>>> bs.n().extto(6) # Error
```

### Bytewise Operations

Operands with `Bs` objects will first be converted into `Bs`. If the lengths don't match, the shorter one will call `shorter_bs.extto(len(longer_bs))` to fit the longer operand's length.

#### Eamples

```python
>>> a = Bs.from_int(0x0a00)
>>> a
<Bs 2: [00, 0a]>

>>> b = Bs.from_int(0x0b)
>>> b
<Bs 1: [0b]>

>>> a + b # b will be unrolled to <Bs 2: [0b, 0b]>
<Bs 2: [0b, 15]>

>>> a + b.n()
# Error: length not matched: (2, 1)

>>> a - b
<Bs 2: [f5, ff]>

>>> a * b
<Bs 2: [00, 6e]>

>>> a / b
<Bs 2: [00, 00]>

>>> a // b
<Bs 2: [00, 00]>

>>> a ** b
<Bs 2: [00, 00]>

>>> a % b
<Bs 2: [00, 0a]>
```

#### Binary operations with other types

The other operand will first be converted into `Bs` as well, with automatic repeating to fit the length of our `Bs`.

```python
>>> cafe = Bs.from_hex('c01dcafe')
>>> cafe
<Bs 4: [c0, 1d, ca, fe]>

>>> cafe + 1
<Bs 4: [c1, 1e, cb, ff]>

>>> 1 + cafe
<Bs 4: [c1, 1e, cb, ff]>

>>> cafe + '糖'
<Bs 4: [a7, d0, 60, e5]>

>>> cafe + b'cafe'
<Bs 4: [23, 7e, 30, 63]>

>>> cafe + b'sugar'
<Bs 5: [33, 92, 31, 5f, 32]>

>>> cafe + [1, 2, 3, 4]
<Bs 4: [c1, 1f, cd, 02]>

>>> cafe + range(5)
<Bs 5: [c0, 1e, cc, 01, c4]>

>>> cafe.p(0) + [0] * 6
<Bs 6: [c0, 1d, ca, fe, 00, 00]>

>>> cafe.bin()
'00000011101110000101001101111111'

>>> (cafe >> 1).bin() # for each byte
'00000110011100001010011011111110'

>>> (cafe << 1).bin() # for each byte
'00000001010111000010100100111111'
```

#### Other useful methods

```python
>>> bs = Bs(range(7))
>>> bs
<Bs 7: [00, 01, 02, 03, 04, 05, 06]>

# Reverse
>>> bs.rev()
<Bs 7: [06, 05, 04, 03, 02, 01, 00]>

# Roll
>>> bs.roll(1)
<Bs 7: [06, 00, 01, 02, 03, 04, 05]>

>>> bs.roll(-1)
<Bs 7: [01, 02, 03, 04, 05, 06, 00]>

>>> bs.rjust(10, 0xff)
<Bs 10: [ff, ff, ff, 00, 01, 02, 03, 04, 05, 06]>

>>> bs.ljust(10, 0xff)
<Bs 10: [00, 01, 02, 03, 04, 05, 06, ff, ff, ff]>

# Iterate over every n bytes
>>> bs.every()
[<Bs 1: [00]>, <Bs 1: [01]>, <Bs 1: [02]>, <Bs 1: [03]>, <Bs 1: [04]>, <Bs 1: [05]>, <Bs 1: [06]>]

>>> bs.every(n=3)
[<Bs 3: [00, 01, 02]>, <Bs 3: [03, 04, 05]>, <Bs 1: [06]>]

>>> bs.every(n=3, m=list) # map
[[0, 1, 2], [3, 4, 5], [6]]

>>> bs.every(n=3, m=int)
[131328, 328707, 6]

>>> bs.every(n=4, m=lambda i: i.asint(32)) # with map
[50462976, 394500]

>>> bs.every(4, list, lambda i: 2 in i) # filter before map
[[0, 1, 2, 3]]

>>> bs.every(4, f=lambda i: 2 in i) # only filter
[<Bs 4: [00, 01, 02, 03]>]
```

### Bitwise Operations

Operating over bits.

#### Basic properties 

```python
#                       v MSB          v LSB
>>> bs = Bs.from_bin('0b1111000011001100')

>>> bs.bin()
'0011001100001111'

>>> bin(bs)
'0b1111000011001100'

>>> bs
<Bs 2: [cc, f0]>

>>> bs.int()
61644

#                     v LSB          v MSB
>>> bs = Bs.from_bin('1111000011001100')

>>> bs.bin()
'1111000011001100'

>>> bin(bs)
'0b11001100001111'

>>> bs
<Bs 2: [0f, 33]>

>>> bs.int()
13071
```

#### Logical operations

```python
>>> x = Bs.from_bin('1111000010101010')

>>> (~x).bin()
'0000111101010101'

>>> y = Bs.from_bin('1' * 16)

>>> (x & y).bin()
'1111000010101010'

>>> (x | y).bin()
'1111111111111111'

>>> (x ^ y).bin()
'0000111101010101'
```

#### Shifting over all bits

```python
>>> bs = Bs.from_bin('1100000000000001')

>>> bs.shift(1).bin()
'0110000000000000'

>>> bs.shift(-1).bin()
'1000000000000010'

>>> bs.asint()
-32765

>>> bs.shift(-1, a=True).bin() # arithmetic
'1000000000000011'

>>> bs.shift(-2, a=True).bin()
'0000000000000111'

>>> bs.shift(-5, a=True).bin()
'0000000000111111'

>>> bs.shift(-100, a=True).bin()
'1111111111111111'

>>> bs = Bs.from_bin('0000000000000010')

>>> bs.asint()
16384

>>> bs.shift(1, a=True).bin()
'0000000000000000'

>>> bs.shift(1, a=True).asint()
0

>>> bs.shift(-1, a=True).bin()
'0000000000000100'

>>> bs.shift(-1, a=True).asint()
8192

>>> bs.shift(-5, a=True).bin()
'0000000001000000'

>>> bs.shift(-5, a=True).asint()
512

>>> bs.shift(-100, a=True).bin()
'0000000000000000'

>>> bs.shift(-100, a=True).asint()
0
```

#### Other useful methods

```python
>>> bs = Bs.from_bin('1100000000000001')

>>> bs.revbits().bin()
'1000000000000011'

>>> bs.rollbits(1).bin()
'1110000000000000'

>>> bs.rollbits(-1).bin()
'1000000000000011'

>>> bs.bits()
['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1']

>>> bs.bits(every=3)
['110', '000', '000', '000', '000', '1']

>>> [b for b in bs.bits(every=3) if len(b) == 3]
['110', '000', '000', '000', '000']
```


### Convertions

#### To integer

```python
>>> bs = Bs(0, 0b10000000)
>>> bs
<Bs 2: [00, 80]>

>>> bs.int()
32768

>>> int(bs)
32768

>>> bs.asint()
-32768

>>> bs.asint(8)
0

>>> bs.asint(16)
-32768

>>> bs.asint(32)
32768

>>> bs.asuint()
32768

>>> bs.asuint(8)
0

>>> bs = Bs.rand(32)
>>> bs
<Bs 32: [51, 2a, aa, dc, 83, 08, 0c, 84, 10, 43, 5f, 5c, db, de, 97, 17, 55, 49, 4e, f3, 89, b3, 45, 03, c1, 98, 77, fc, 90, bd, 50, 6b]>

>>> bs.asints(32)
[-592827823, -2079586173, 1549746960, 395828955, -212973227, 54899593, -59270975, 1800453520]

>>> bs.asuints(32)
[3702139473, 2215381123, 1549746960, 395828955, 4081994069, 54899593, 4235696321, 1800453520]
```

#### To string

```python
>>> bs = Bs('las vegas')
>>> bs
<Bs 9: [6c, 61, 73, 20, 76, 65, 67, 61, 73]>

>>> bs.str()
'las vegas'

>>> str(bs)
'las vegas'

>>> bs.hex()
'6c6173207665676173'

>>> hex(bs)
'0x73616765762073616c'

>>> oct(bs)
'0o346605473127304034660554'

>>> bs.bin()
'001101101000011011001110000001000110111010100110111001101000011011001110'

>>> bin(bs)
'0b11100110110000101100111011001010111011000100000011100110110000101101100'
```

### Other

#### Base64 encode

```python

>>> Bs('Las Vegas').base64()
'TGFzIFZlZ2Fz'
```

#### Hashing

```python
>>> bs = Bs('Las Vegas')

>>> bs.hash('md5')
<Bs 16: [05, c2, 7b, f0, 09, 32, 57, 2d, e2, 8b, f6, 5a, 05, 39, ba, 97]>

>>> bs.hash('md5').hex()
'05c27bf00932572de28bf65a0539ba97'

>>> bs.hash('sha256')
<Bs 32: [2b, d2, 5c, d9, 60, ab, a8, b7, 06, e2, b6, 7f, 2b, b3, 8b, 75, 0e, e5, 38, 4b, 0e, 98, 83, 05, 3e, bc, 3b, 89, ef, 4d, de, f9]>

>>> bs.hash('sha256').hex()
'2bd25cd960aba8b706e2b67f2bb38b750ee5384b0e9883053ebc3b89ef4ddef9'

# See what's available
>>> import hashlib
>>> hashlib.algorithms_guaranteed
{'sha384', 'shake_128', 'sha3_256', 'sha3_512', 'md5', 'sha512', 'shake_256', 'sha3_384', 'sha1', 'sha3_224', 'blake2b', 'blake2s', 'sha256', 'sha224'}
```

#### Command pipe

```python
>>> Bs('stdin input').pipe('tr a-z A-Z').str()
'STDIN INPUT'

>>> Bs('stdin input').pipe('base64').pipe('tee /dev/stderr').pipe('base64 --decode').str()
c3RkaW4gaW5wdXQ=
'stdin input'

>>> print(Bs().pipe('ls').str())
LICENSE
README.md
bites
bites.egg-info
build
dist
setup.py

```

#### IPv4

```python
>>> ip = Bs.from_ip4('192.168.1.1/16')
>>> ip
<Bs 4: [00, 00, a8, c0]>

>>> ip.every(1, int)
[0, 0, 168, 192]

>>> ip.ip4()
'192.168.0.0'
```
