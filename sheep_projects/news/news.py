import os.path, pickle, time

cache_name = "/tmp/news.cache"
cache_time = 3600 * 6

"""
---------------------
please call this:
    news = get_news()
---------------------
news = {
    "time": TIME,
    "data": [
        {
            "title": TITLE,
            "content": CONTENT
        },
        ...
    ]
}
---------------------
this function will save cache at /tmp/news.cache
---------------------
"""
def get_news():
    data = read_cache()
    if not data:
        data = get_data()
    return data

#########################################
# PRIVATE                               #
#########################################

def main():
    print(get_news())

def get_data():
    from requests import get
    from scrapy import Selector
    base = "http://m.ltn.com.tw/"
    newspaper = base + "newspaper/"
    urls = map(lambda x: base + x, Selector(text=get(newspaper).text).re("news/focus/paper/\d+"))
    data = {"data": [], "time": time.time()}
    for url in urls:
        print("parsing:", url)
        s = Selector(text=get(url).text)
        title = s.css(".com [data-desc=內容頁] *::text").extract()[0]
        content = "\n".join(map(lambda x: dtag(x), s.css('.com [data-desc=內文] *').extract()))
        data["data"].append({
            "title": title,
            "content": content
        })
    write_cache(data)
    return data

def read_cache():
    if not os.path.isfile(cache_name):
        return None
    with open(cache_name, "rb") as f:
        cache = pickle.load(f)
        if time.time() - cache["time"] > cache_time:
            from threading import Thread
            Thread(target=get_data).start()
        return cache["data"]
    return None

def dtag(s):
    import re
    return re.sub(re.compile('<.*?>'), '', s)

def write_cache(data):
    cache = {"time": time.time(), "data": data}
    with open(cache_name, "wb") as f:
        pickle.dump(cache, f)

if __name__ == "__main__":
    main()