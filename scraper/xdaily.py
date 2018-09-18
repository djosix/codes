import sys

try:
    url = sys.argv[1]
    try:
        path = sys.argv[2]
    except:
        path = '.'
except:
    print('USAGE: {} <url> [save_dir]'.format(sys.argv[0]))
    exit()

from requests import Session
from scrapy.selector import Selector
import os

sess = Session()

def save_img(img, path):
    r = sess.get(img, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    else:
        print('Failed')

try:
    os.mkdir(path)
except:
    pass

for i, img in enumerate(Selector(text=sess.get(url).text).css('img').xpath('@data-original').extract()):
    ext = img[img.rfind('.'):]
    dest = os.path.join(path, '{:03d}{}'.format(i, ext))
    print('Downloading {} {} {}'.format(i, img, dest))
    save_img(img, dest)

