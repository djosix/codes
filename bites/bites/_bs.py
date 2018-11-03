import collections
import itertools
import base64
import hashlib
import random
import subprocess
import urllib.parse


class Bs:
    @classmethod
    def from_base64(cls, b64):
        return cls(base64.b64decode(b64))

    @classmethod
    def from_int(cls, n):
        assert n >= 0, 'required a positive value'
        data = []
        while n:
            data.append(n & 0xff)
            n >>= 8
        return cls(data)

    @classmethod
    def from_hex(cls, hs):
        if hs.startswith('0x'):
            return cls.from_int(int(hs, 16))
        else:
            n = len(hs) + -len(hs) % 2
            hs = hs.ljust(n, '0')
            return cls(int(hs[i:i+2], 16) for i in range(0, len(hs), 2))

    @classmethod
    def from_bin(cls, bs):
        if bs.startswith('0b'):
            return cls.from_int(int(bs, 2))
        else:
            nb = (len(bs) + 7) // 8
            return cls.from_int(int(bs[::-1], 2)).ljust(nb, 0)

    @classmethod
    def from_ip4(cls, ip):
        ip, *mask = ip.split('/')
        nums = [int(n) for n in ip.split('.')]
        assert len(nums) == 4, 'an IP address requires 4 integers.'
        bs = Bs(nums).rev()
        if len(mask):
            mask = list(map(int, filter(len, mask)))
            assert len(mask) == 1, 'invalid mask format.'
            mask = mask[0]
            if mask == 0:
                return Bs(0, 0, 0, 0)
            if mask == 32:
                return bs
            assert 0 < mask < 32, 'invalid mask value.'
            return Bs.from_bin(bs.bin()[-mask:].rjust(32, '0')).padto(4)
        return bs
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(f.read())
    
    @classmethod
    def rand(cls, n, cs=range(256)):
        return cls(random.choices(cs, k=n))

    def __init__(self, *objs, p=None, r=True):
        self._p = p
        self._r = r
        self.data = []
        for obj in objs:
            if isinstance(obj, collections.Iterable):
                for item in obj:
                    self._additem(item)
            else:
                self._additem(obj)
        self._correct()
    
    def __iter__(self):
        yield from self.data    
    def __getitem__(self, i):
        return type(self)(self.data[i])
    
    def __setitem__(self, i, n):
        if isinstance(i, slice):
            r = range(*i.indices(len(self)))
            n = n if type(n) is type(self) else type(self)(n)
            if len(r) > len(n) and n._r:
                n = n.rep((len(r) + len(n) - 1) // len(n))
            elif len(r) < len(n) and n._r:
                n = n[:len(r)]
            elif len(r) > len(n) and n._p is not None:
                n = n.ljust(len(r), n._p)
            else:
                assert len(r) == len(n), \
                        'cannot set values to {}: r={}, p={}, l={}'.format(
                            r, n._r, n._p, len(n))
            for j, v in zip(r, n):
                self[j] = v
        else:
            n = type(self)(n)
            assert len(n) == 1, 'expected only one value'
            self.data[i] = n[0]
    
    def __repr__(self):
        to_hex = lambda b: hex(b)[2:].rjust(2, '0')
        btext = ', '.join(map(to_hex, self[:128]))
        btext += ', ...' if len(self) > 128 else ''
        return '<{} {}: [{}]>'.format(type(self).__name__, len(self), btext)
    
    def __str__(self):
        return self.str()
    
    def __bytes__(self):
        return self.bytes()
    
    def __int__(self):
        return self.int()
    
    def __index__(self):
        return self.int()
    
    def __hash__(self):
        return self.int()

    def __bool__(self):
        return bool(self.int())
    
    def __len__(self):
        return self.len()
    
    def __neg__(self):
        return type(self)(-i for i in self)
    
    def __add__(self, obj):
        return self._operate(obj, lambda x, y: x + y)
    
    def __sub__(self, obj):
        return self + -obj
    
    def __mul__(self, obj):
        return self._operate(obj, lambda x, y: x * y)
    
    def __pow__(self, obj):
        return self._operate(obj, lambda x, y: x ** y)
    
    def __mod__(self, obj):
        return self._operate(obj, lambda x, y: x % y)

    def __truediv__(self, obj):
        return self._operate(obj, lambda x, y: x // y)
    
    def __floordiv__(self, obj):
        return self._operate(obj, lambda x, y: x // y)
    
    def __and__(self, obj):
        return self._operate(obj, lambda x, y: x & y)
    
    def __or__(self, obj):
        return self._operate(obj, lambda x, y: x | y)
    
    def __xor__(self, obj):
        return self._operate(obj, lambda x, y: x ^ y)
    
    def __invert__(self):
        return type(self)(~i for i in self)
    
    def __lshift__(self, obj):
        return self._operate(obj, lambda x, y: x << y)
    
    def __rshift__(self, obj):
        return self._operate(obj, lambda x, y: x >> y)

    def __matmul__(self, obj):
        tmp = type(self)(self)
        tmp._additem(obj)
        tmp._correct()
        return tmp

    def __radd__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x + y)
    
    def __rsub__(self, obj):
        return obj + -self
    
    def __rmul__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x * y)
    
    def __rpow__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x ** y)

    def __rmod__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x % y)

    def __rtruediv__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x // y)
    
    def __rfloordiv__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x // y)
    
    def __rand__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x & y)
    
    def __ror__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x | y)
    
    def __rxor__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x ^ y)
    
    def __rlshift__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x << y)
    
    def __rrshift__(self, obj):
        return type(self)(obj)._operate(self, lambda x, y: x >> y)

    def __rmatmul__(self, obj):
        tmp = type(self)(obj)
        tmp._additem(self)
        tmp._correct()
        return tmp
    
    def __eq__(self, obj):
        return self.data == type(self)(obj).data
    
    def __lt__(self, obj):
        return self.int() < type(self)(obj).int()
    
    def __gt__(self, obj):
        return self.int() > type(self)(obj).int()
    
    def __le__(self, obj):
        return self.int() <= type(self)(obj).int()
    
    def __ge__(self, obj):
        return self.int() >= type(self)(obj).int()
    
    def __ne__(self, obj):
        return not (self == obj)

    def ljust(self, width, fill=0):
        return type(self)(bytes(self.data).ljust(width, bytes([fill])))
    
    def rjust(self, width, fill=0):
        return type(self)(bytes(self.data).rjust(width, bytes([fill])))
    
    def rep(self, n):
        '''Repeat the bytes n times.
        '''
        return type(self)(self.data * n)
    
    def repto(self, n):
        '''Repeat the bytes to a specific length.
        '''
        return type(self)(self.data[i % len(self)] for i in range(n))
    
    def padto(self, n, p=0):
        '''Pad the bytes with `p` to a specific length.
        '''
        if n <= len(self):
            return self[:n]
        return type(self)(self, type(self)(p).repto(n - len(self)))
    
    def extto(self, n):
        '''Extend the bytes to a specific length:
        
        - if `bs._p` is set, return `bs.padto(n, bs._p)`
        - else if `bs._r` is true, return `bs.repto(n)`
        - else raise error
        '''
        if self._r:
            return self.repto(n)
        if self._p is not None:
            return self.padto(n, self._p)
        assert False, 'padding or repeating not allowed'
    
    def rev(self):
        return type(self)(reversed(self))
    
    def revbits(self):
        return type(self).from_bin(self.bin()[::-1])
    
    def roll(self, n):
        if n == 0:
            return type(self)(self)
        n = -n % len(self)
        return type(self)(self.data[n:], self.data[:n])
    
    def rollbits(self, n):
        if n == 0:
            return type(self)(self)
        n = -n % (len(self) * 8)
        bits = self.bin()
        return type(self).from_bin(bits[n:] + bits[:n])
    
    def shift(self, nbits, a=False):
        allbits = self.len() << 3
        if nbits == 0:
            return type(self)(self)
        elif nbits > 0:
            op = lambda x: (x << nbits) % allbits
        elif nbits < 0:
            op = lambda x: x >> -nbits
        if a:
            o = 1 << allbits
            s = ~o % o
            if self.data[-1] & 0b10000000:
                k = min(0, max(nbits, -(allbits - 1)))
                f = 1 << (allbits + k - 1)
                m = ~f % f
                s ^= m
                n = op(self.int()) & m | s
            else:
                f = 1 << (allbits - 1)
                m = ~f % f
                n = op(self.int()) & m
            return type(self).from_int(n).padto(len(self))
        return type(self).from_int(op(self.int())).padto(len(self))
    
    def rot13(self):
        return self.rot(13, rs=((97, 123), (65, 91)))

    def rot(self, n, rs=((0, 256),)):
        data = self.data.copy()
        key = Bs(n).repto(self.len())
        for i, c in enumerate(data):
            for r0, r1 in rs:
                if r0 <= c < r1:
                    data[i] = (c - r0 + key[i]) % (r1 - r0) + r0
        return type(self)(data)

    def append(self, obj):
        self._additem(obj)
        self._correct()

    def n(self):
        return type(self)(self, p=None, r=False)

    def p(self, p):
        return type(self)(self, p=p, r=False)
    
    def r(self):
        return type(self)(self, p=None, r=True)

    def len(self):
        return len(self.data)

    def str(self):
        return bytes(self.data).decode()

    def bytes(self):
        return bytes(self.data)

    def int(self):
        p, value = 1, 0
        for n in self.data:
            value += n * p
            p <<= 8
        return value
    
    def list(self):
        return self.data.copy()
    
    def tuple(self):
        return tuple(*self.data)
    
    def asint(self, nbits=None, le=True):
        if nbits is None:
            nbits = self.len() << 3
        val = self.asuint(nbits, le)
        sbit = 1 << (nbits - 1)
        if val & sbit:
            return (val & ~sbit) - sbit
        return val & ~sbit
    
    def asuint(self, nbits=None, le=True):
        if nbits is None:
            nbits = self.len() << 3
        assert nbits % 8 == 0, 'invalid bits: {}'.format(nbits)
        nbytes = nbits // 8
        val = self[:nbytes].int() if le else self[:nbytes].rev().int()
        return val
    
    def asints(self, n, le=True):
        bss = self.every(n // 8, type(self))
        return [bs.asint(n) for bs in bss]
    
    def asuints(self, n, le=True):
        bss = self.every(n // 8, type(self))
        return [bs.asuint(n) for bs in bss]

    def bin(self):
        return ''.join(bin(n)[2:].rjust(8, '0')[::-1] for n in self)
    
    def hex(self):
        return ''.join(hex(n)[2:].rjust(2, '0') for n in self)
    
    def ip4(self):
        assert self.len() == 4
        return '.'.join(map(str, self.rev().every(1, int)))
    
    def base64(self):
        return base64.b64encode(bytes(self)).decode()
    
    def urlenc(self):
        return ''.join(self.every(1, lambda c: '%{}'.format(c.hex())))
    
    def quote(self, plus=False):
        quoted = urllib.parse.quote_from_bytes(self.bytes())
        if plus:
            return quoted.replace('%20', '+')
        return quoted
    
    def uenc(self):
        fmt = lambda u: '\\U{u:08x}' if u >> 16 else '\\u{u:04x}'
        return ''.join(map(fmt, map(ord, self.str())))

    def xenc(self):
        return ''.join(map('\\x{:02x}'.format, self.bytes()))
    
    def hash(self, name):
        assert name in hashlib.algorithms_guaranteed, \
            'hash `{}` is not available'.format(name)
        hasher = getattr(hashlib, name)()
        hasher.update(self.bytes())
        return type(self)(hasher.digest())

    def bits(self, every=1):
        bs = self.bin()
        return [bs[i:i+every] for i in range(0, len(bs), every)]
    
    def nbits(self):
        return self.len() << 3
    
    def every(self, n=1, m=lambda x: x, f=lambda x: True):
        # Wow!
        return list(
            map(
                m,
                filter(
                    f,
                    map(
                        lambda i: self[i:i+n],
                        range(
                            0,
                            len(self),
                            n
                            )))))

    def split(self, s=None, m=lambda i: i, f=lambda i: True):
        if s is None:
            bss = self.bytes().split()
        else:
            bss = self.bytes().split(Bs(s).bytes())
        bss = [Bs(bs) for bs in bss]
        bss = list(map(m, filter(f, (Bs(bs) for bs in bss))))
        return bss

    def dump(self, path):
        with open(path, 'wb') as f:
            f.write(self.bytes())

    def pipe(self, command, **kw):
        kw.setdefault('shell', True)
        kw.setdefault('stdout', subprocess.PIPE)
        output = subprocess.run(command, input=self.bytes(), **kw).stdout
        return type(self)(output)

    def _correct(self):
        for i in range(len(self)):
            self.data[i] %= 256

    def _additem(self, item):
        assert isinstance(item, (int, collections.Iterable)), \
                'unexpected type: {}'.format(type(item))
        if isinstance(item, int):
            self.data.append(item)
        elif isinstance(item, str):
            self.data += list(item.encode())
        else:
            for i in item:
                self._additem(i)

    def _operate(self, obj, func):
        obj = obj if isinstance(obj, type(self)) else type(self)(obj)
        if len(self) == len(obj):
            x, y = self, obj
        elif len(self) > len(obj) and obj._p is not None:
            x, y = self, obj.ljust(len(self), obj._p % 256)
        elif len(self) < len(obj) and self._p is not None:
            x, y = self.ljust(len(obj), self._p % 256), obj
        elif len(self) > len(obj) and obj._r:
            x, y = self, obj.rep(len(self) // len(obj) + 1)[:len(self)]
        elif len(self) < len(obj) and self._r:
            x, y = self.rep(len(obj) // len(self) + 1)[:len(obj)], obj
        else:
            assert False, 'length not matched: {}'.format((len(self), len(obj)))
        return type(self)(func(i, j) for i, j in zip(x, y))
