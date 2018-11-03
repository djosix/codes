# Requirement
#   pip3 install pypinyin
# Examples
#   python3 pinyin.py -i A11_0_ori.wav.trn
#   python3 pinyin.py -i A11_0_ori.wav.trn -o A11_0.wav.trn

DICT_NAME = 'lexicon.txt'

import pypinyin, argparse

with open(DICT_NAME) as f:
    l = [i.split() for i in f.readlines()]
    d = {i[0]: i[1:] for i in l}

def 羅馬拼音(s):
    l = pypinyin.pinyin(s, pypinyin.TONE3)
    f = lambda i: ''.join(i).strip() != ''
    return ' '.join(' '.join(i) for i in filter(f, l))

def 漢語拼音(s):
    l = [d.get(c, []) for c in s]
    f = lambda i: ''.join(i).strip() != ''
    return ' '.join(' '.join(i) for i in filter(f, l))

if __name__ == '__main__':
    a = argparse.ArgumentParser(description='拼音轉換器')
    a.add_argument('-i', '--input', help='input file', required=True)
    a.add_argument('-o', '--output', help='output file')
    a = a.parse_args()

    i_name = a.input
    o_name = a.output or a.input + '.parsed'

    with open(i_name) as f:
        s = f.read()
    
    s1 = 羅馬拼音(s)
    s2 = 漢語拼音(s)

    with open(o_name, 'wt') as f:
        f.write('\n'.join([s, s1, s2]))

