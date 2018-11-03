import pickle, readline

data = pickle.load(open('data.pkl', 'rb'))
data = [i[0] for user in data.values() for i in user]
data = list(set(data))

try:
    labeled = pickle.load(open('labeled.pkl', 'rb'))
except:
    labeled = {name: False for name in data}
    

def mark(keyword, target, overwrite=True):
    global labeled
    count = 0
    for name, mark in labeled.items():
        if keyword in name:
            if not mark or overwrite:
                labeled[name] = target
                count += 1
    nones = len([mark for mark in labeled.values() if not mark])
    print(count, 'columns affected')
    print(nones, 'nones')

#luis = pickle.load(open('luis.pkl', 'rb'))
#for label, words in luis.items():
#    for word in words:
#        mark(word, label)

def loop():
    while True:
        label, keyword  = str(input()).split()
        mark(keyword, label)

