import os, time, collections, random, itertools, concurrent, json, functools, re
import numpy as np
from IPython import embed
from sys import exit

class Node:
    slots = ['Webpage', 'LeafIndex', 'ContentId', 'TypeSetId',
             'PTypeSetId', 'PathId', 'SimTECId', 'Content']
    
    null_score = -0.1
    similarity_scores = np.array([2.0, 1.0, 0.5, 2.0, 1.0])

    def __init__(self, *args, **kwargs):
        assert len(args) == len(Node.slots) or \
            (not args and set(kwargs.keys()) == set(Node.slots))
        for slot, arg in zip(Node.slots, args):
            setattr(self, slot, arg)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.array = np.array([hash(arg) for arg in args[2:-1]])

    def __repr__(self):
        return 'Node({})'.format(
            ', '.join(
                '{}={}'.format(slot, repr(getattr(self, slot)))
                for slot in self.slots))
    
    def __str__(self):
        return 'N-{}-{}'.format(self.Webpage, self.LeafIndex) # pylint: disable=E1101

    @classmethod
    def similarity(cls, n1, n2):
        # n1 and n2 are 1-D arrays
        cond = int(n1 is None) + int(n2 is None)
        if cond:
            return cond * cls.null_score
        return cls.similarity_scores[n1.array == n2.array].sum()


class Utils:

    @staticmethod
    def txt_file_paths_under(root):
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.txt'):
                    yield os.path.join(dirpath, filename)

    @staticmethod
    def extract_npages(path):
        return int(re.search(r'.+-(\d+)\.txt', os.path.basename(path)).groups()[0])

    @staticmethod
    def load_segs(path):
        with open(path) as f:
            lines = f.readlines()
        segs = [[]]
        for line in lines[1:]:
            line = line.strip()
            if len(line):
                node = Node(*line.split('\t'))
                segs[-1].append(node)
            else:
                segs.append([])
        return [seg for seg in segs if len(seg)]

    @staticmethod
    def seg2pages(seg, n):
        pages = collections.OrderedDict()
        for node in seg:
            page = node.Webpage
            if page not in pages:
                pages[page] = []
            pages[page].append(node)
        pages = list(pages.values())
        for _ in range(n - len(pages)):
            pages.append([])
        return pages
    
    @staticmethod
    def uproduct(*generators):
        '''Non-recursive DFS generator for unique element products.'''
        lists = list(map(list, generators))
        n = min(len(lists), len(set(i for l in lists for i in l)))
        stack = []
        indeces = [0] * n
        while True:
            i = len(stack)
            if i < n:
                if indeces[i] == len(lists[i]):
                    if i == 0:
                        return
                    indeces[i] = 0
                    stack.pop()
                else:
                    item = lists[i][indeces[i]]
                    if item not in stack:
                        stack.append(item)
                    indeces[i] += 1
            else:
                # print(tuple(stack))
                yield tuple(stack)
                stack.pop()

    @staticmethod
    def row_similarity(r1, r2):
        '''Similarity score of two rows.'''
        return sum(Node.similarity(n1, n2) for n1, n2 in zip(r1, r2))

    @staticmethod
    def rows_score(rows):
        '''Cross-similarity score sum of all rows.'''
        combinations = itertools.combinations(rows, 2)
        return sum(Utils.row_similarity(r1, r2) for r1, r2 in combinations)

    @staticmethod
    def similarity_matrix(nodes1, nodes2):
        '''Fast similarity matrix calculation using numpy.'''
        n1, n2 = len(nodes1), len(nodes2)
        array1 = np.stack([node.array for node in nodes1]) # [n1, s]
        array2 = np.stack([node.array for node in nodes2]) # [n2, s]
        array1 = np.expand_dims(array1, 1) # [n1, 1, s]
        array2 = np.expand_dims(array2, 0) # [1, n2, s]
        identical = (array1 == array2) # [n1, n2, s]
        scores = np.tile(Node.similarity_scores, (n1, n2, 1)) # [n1, n2, s]
        matrix = (scores * identical).sum(-1) # [n1, n2]
        return matrix


class BeamSearch:

    State = collections.namedtuple('State', ['row', 'score'])

    @staticmethod
    def heuristic_row_generator(history, next_nodes, width):
        if not next_nodes:
            yield [None] * width
            return
        matrix = np.zeros((width, len(next_nodes)))
        for state in history:
            nodes, indeces = [], []
            for i, node in enumerate(state.row):
                if node is not None:
                    indeces.append(i)
                    nodes.append(node)
            simmat = Utils.similarity_matrix(nodes, next_nodes)
            for i, arr in enumerate(simmat):
                matrix[indeces[i]] += arr
        # embed()
        argsorted = (-matrix).argsort() # high to low
        orders = []
        for nnis in argsorted.T:
            col = {nni: [] for nni in set(nnis)}
            for ri, nni in enumerate(nnis):
                col[nni].append(ri)
            orders += sorted(col.items(), key=lambda t: len(t[1]))
        for begin in range(len(orders)):
            row = [None] * width
            nn_used = [False] * len(next_nodes)
            for nni, ris in orders[begin:]:
                if nn_used[nni]:
                    continue
                for ri in ris:
                    if row[ri] is None:
                        row[ri] = next_nodes[nni]
                        nn_used[nni] = True
                        break
            if not all(nn_used):
                return
            yield row

    @staticmethod
    def randbest_row_generator(history, next_nodes, width, buffer=128):
        if not next_nodes:
            yield [None] * width
            return
        matrix = np.zeros((width, len(next_nodes)))
        for state in history:
            nodes, indeces = [], []
            for i, node in enumerate(state.row):
                if node is not None:
                    indeces.append(i)
                    nodes.append(node)
            simmat = Utils.similarity_matrix(nodes, next_nodes)
            for i, arr in enumerate(simmat):
                matrix[indeces[i]] += arr
        def irow_score(irow):
            gen = ((i, nt[0]) for i, nt in enumerate(irow) if nt is not None)
            ri, ni = zip(*gen)
            return matrix[ri, ni].sum()
        def shuffled(items):
            items = items.copy()
            random.shuffle(items)
            return items
        irow = list(enumerate(next_nodes)) + [None] * (width - len(next_nodes))
        while True:
            with concurrent.futures.ThreadPoolExecutor(buffer) as tpe:
                irows = tpe.map(shuffled, [irow] * buffer)
            yield [
                nt[1] if nt is not None else None
                for nt in max(irows, key=irow_score)]

    @staticmethod
    def random_row_generator(history, next_nodes, width):
        row = next_nodes + [None] * (width - len(next_nodes))
        while True:
            random.shuffle(row)
            yield row

    def __init__(self, beam_size, row_generator, shuffle=False):
        self.beam_size = beam_size
        self.row_generator = row_generator
        self.shuffle = shuffle
    
    def __call__(self, pages):
        return self.search(pages)

    def next_states(self, history, next_nodes, width):
        '''Find next states given beam history.'''
        next_states = []
        for row in self.row_generator(history, next_nodes, width):
            score = sum(Utils.row_similarity(state.row, row) for state in history)
            next_states.append(self.__class__.State(row, history[-1].score + score))
            if len(next_states) == self.beam_size:
                break
        return next_states

    def search(self, pages):
        if self.shuffle:
            pages = pages.copy()
            for i, page in enumerate(pages):
                pages[i] = page.copy()
                random.shuffle(pages[i])

        width = max(map(len, pages)) # max width of rows
        init_nodes, *the_rest = pages
        init_row = init_nodes + [None] * (width - len(init_nodes))
        init_state = self.__class__.State(init_row, 0)
        init_history = [init_state]
        beams = [init_history]
        for nodes in the_rest:
            next_beams = []
            for history in beams:
                for next_state in self.next_states(history, nodes, width):
                    next_beams.append(history + [next_state])
            next_beams.sort(key=lambda history: -history[-1].score)
            beams = next_beams[:self.beam_size]
        best_history = beams[0]
        return [state.row for state in best_history]

def no_search(pages):
    '''Do nohing (just padding).'''
    rows = []
    width = max(map(len, pages))
    for page in pages:
        rows.append(page + [None] * (width - len(pages)))
    return rows

def random_search(pages):
    rpages = []
    for page in pages:
        page = page.copy()
        random.shuffle(page)
        rpages.append(page)
    return no_search(rpages)

def test(path, search_funcs={'none': no_search}):
    print(f'[{path}]')

    segs = Utils.load_segs(path)
    depth = Utils.extract_npages(path)
    result = {'path': path, 'segs': []}

    for i, seg in enumerate(segs):
        print(f'  segment #{i}')

        pages = Utils.seg2pages(seg, depth)
        result['segs'].append({})

        for func_name, func in search_funcs.items():
            start_time = time.time()
            rows = func(pages)
            duration = (time.time() - start_time)
            score = Utils.rows_score(rows)
            result['segs'][-1][func_name] = (score, duration)
            print(f'    {func_name}: {(score, duration)}')
    
    avg = {fn: [] for fn in search_funcs}
    for d in result['segs']:
        for fn, tup in d.items():
            avg[fn].append(tup)
    print('  average')
    for fn, tups in avg.items():
        avg[fn] = tuple(map(lambda s: sum(s) / len(tups), zip(*tups)))
        print(f'    {fn}: {avg[fn]}')
    result['avg'] = avg

    return result


if __name__ == '__main__':
    search_funcs = {
        'none': no_search,
        'random': random_search}
    for i in [1, 2, 3, 5, 7, 11]:
        search_funcs[f'heuristic-beam-{i}'] = BeamSearch(
            i, BeamSearch.heuristic_row_generator, shuffle=True)
    # for i in [8, 16, 64, 128, 256]:
    #     search_funcs[f'random-beam-{i}'] = BeamSearch(
    #         i, BeamSearch.random_row_generator)
    # for i in [8, 16, 64, 128, 256]:
    #     for j in [8, 16, 64, 128]:
    #         row_generator = functools.partial(BeamSearch.randbest_row_generator, buffer=j)
    #         search_funcs[f'randbest-{j}-beam-{i}'] = BeamSearch(i, row_generator)
    
    test_func = functools.partial(test, search_funcs=search_funcs)
    # results = list(map(test_func, Utils.txt_file_paths_under('./data')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ppe:
        results = list(ppe.map(test_func, Utils.txt_file_paths_under('./data')))

    with open('result.json', 'w') as f:
        json.dump(results, f)

    with open('result.txt', 'w') as f:
        for result in results:
            path = result['path']
            f.write(f'[{path}]\n')
            for func, tup in result['avg'].items():
                f.write(f'  {func}: {tup}\n')
