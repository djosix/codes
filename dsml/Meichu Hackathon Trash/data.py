import pandas as pd, re, tqdm

data = {}

print('Reading csv data...')
rows = pd.read_csv('invoice20170411.csv').values[:, [1, 7, 8]]

for uid, str1, str2 in tqdm.tqdm(rows):
    try:
        str1 = str1.replace('\n', '').replace('\r', '')
        str2 = str2.replace('\n', '').replace('\r', '')
        line = re.sub('^.*==[:\*]*(\d+:\d+:)?', '', str1 + re.sub('^\*+', '', str2))

        e = re.search('^([012]):', line)
        if e is None:
            continue

        enc = e.group(1)

        ilist = list(filter(lambda m: m[0] and re.match('^\d+$', m[0]) is None, map(lambda m: m.split(':'), re.findall('[^:]+:\d+:\d+', line))))

        if not ilist:
            continue

        for name, number, price in ilist:
            try:
                name = re.sub('[^\u4E00-\u9fa5]', ' ', name)
                name = re.sub(' +', ' ', name).strip()

                if not name:
                    continue

                if int(enc) == 0:
                    break

                name = name.strip()

                if not name:
                    continue
                
                r = (name, float(number), float(price))
                # print(r)
                data.setdefault(int(uid), []).append(r)
            except:
                continue
    except:
        continue


import pickle
pickle.dump(data, open('data.pkl', 'wb'))
