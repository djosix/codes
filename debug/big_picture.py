
def show(data, depth=0, max_items=5, use_repr=True):
    print('  ' * depth, end='')
    if isinstance(data, list):
        size = len(data)
        types = set(type(i) for i in data)
        get_name = lambda obj: obj.__name__
        set_str = '{' + ', '.join(map(get_name, types)) + '}'
        print('list: {} of {}'.format(size, set_str))
        for i in data[:max_items]:
            show(i, depth=(depth + 1))
        size_remain = len(data[max_items:])
        if size_remain:
            show('... {} more'.format(size_remain),
                 depth=(depth + 1), use_repr=False)
    elif isinstance(data, tuple):
        types = [type(i) for i in data]
        get_name = lambda obj: obj.__name__
        type_str = ', '.join(map(get_name, types))
        print('tuple: {}'.format(type_str))
        for i in data[:max_items]:
            show(i, depth=(depth + 1))
        size_remain = len(data[max_items:])
        if size_remain:
            show('... {} more'.format(size_remain),
                 depth=(depth + 1), use_repr=False)
    elif isinstance(data, dict):
        size = len(data)
        get_name = lambda obj: obj.__name__
        key_types = set(type(k) for k in data.keys())
        key_t_str = '{' + ', '.join(map(get_name, key_types)) + '}'
        val_types = set(type(v) for v in data.values())
        val_t_str = '{' + ', '.join(map(get_name, val_types)) + '}'
        print('dict: {} of {} => {}'.format(
            size, key_t_str, val_t_str))
        items = list(data.items())
        size_items = len(items)
        for i, item in enumerate(items):
            if i < max_items:
                show(item, depth=(depth + 1))
            else:
                size_remain = size_items - max_items
                show('... {} more'.format(size_remain),
                    depth=(depth + 1), use_repr=False)
                break
    else:
        print(repr(data) if use_repr else data)
        

def mergedup(seq):
    result = []
    for item in seq:
        if result and result[-1][0] == item:
            result[-1][1] += 1
        else:
            result.append([item, 1])
    return result


def typetree(data, depth=0):
    global _last_type
    indent = '  '
    typestr = type(data).__name__
    try:
        print(indent * depth + typestr, len(data))
    except:
        print(indent * depth + typestr)
    if isinstance(data, dict):
        typeset = set()
        for key, val in data.items():
            typepair = (type(key), type(val))
            if typepair not in typeset:
                typeset.add(typepair)
                typetree(key, depth + 1)
                typetree(val, depth + 1)
            else:
                print(indent * (depth + 1) + '...')
    elif isinstance(data, list):
        for item in data:
            typetree(item, depth + 1)
    elif isinstance(data, tuple):
        for item in data:
            typetree(item, depth + 1)
