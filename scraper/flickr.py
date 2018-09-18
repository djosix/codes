from splinter import Browser
import time, re, os

b = Browser('chrome')
base = 'https://www.flickr.com'
pages = [
]

def find(attr, html):
    return re.findall(attr + '="(.+?)"', html)

for page in pages:
    b.visit(page)
    links = b.find_by_css('.view.photo-list-photo-view.awake .interaction-view .photo-list-photo-interaction a.overlay')
    html = ''.join(link.outer_html for link in links)
    urls = find('href', html)

    for url in urls:
        b.visit(base + url)
        b.find_by_css('.view.photo-notes-scrappy-view')[0].click()
        img = b.find_by_css('.zoom-photo-container img.zoom-large')[0]
        src = 'https:' + find('src', img.outer_html)[0]
        os.system('wget -P save ' + src)

