
class Apriori:
    def __init__(self, transactions, min_support, verbose=False):
        universe = set()
        for transaction in transactions:
            universe = universe.union(transaction)
        self.transactions = transactions
        self.universe = universe
        self.min_support = min_support
        self.frequent_sets = {}
        self.depth = 0
        self.verbose = verbose

    def run(self):
        k = self.depth + 1
        self.frequent_sets[k] = [frozenset({item}) for item in self.universe]

        while True:
            self.frequent_sets[k] = {items for items in self.frequent_sets[k]
                                     if self._support(items) >= self.min_support}
            
            if len(self.frequent_sets[k]) > 0:
                self.depth += 1
                yield self.frequent_sets[k]
                self._verbose()
            else:
                break

            k = self.depth + 1

            self.frequent_sets[k] = set()
            for i1 in self.frequent_sets[k - 1]:
                for i2 in self.frequent_sets[1]:
                    items = frozenset(i1 | i2)
                    if len(items) == k:
                        self.frequent_sets[k].add(items)

            if self.depth > 1:
                del self.frequent_sets[self.depth]

    def _support(self, items):
        return sum(items.issubset(t) for t in self.transactions)

    def _verbose(self):
        if self.verbose > 0:
            for items in self.frequent_sets[self.depth]:
                set_str = ', '.join(sorted(items))
                support = self._support(items)
                print('depth=%d support=%d { %s }'
                      % (self.depth, support, set_str))


class RuleIterator:
    def __init__(self, apriori, min_confidence=0, verbose=False):
        assert type(apriori) == Apriori
        self.apriori = apriori
        self.verbose = verbose
        self.frequent_sets = []
        self.min_confidence = min_confidence

    def run_apriori(self):
        for result in self.apriori.run():
            self.frequent_sets += [result]
            yield result

    def run(self):
        depth = len(self.frequent_sets)
        for i in reversed(range(depth)):
            for j in reversed(range(i+1)):
                for a in self.frequent_sets[i]:
                    for b in self.frequent_sets[j]:
                        if b & a == set():
                            confidence = self._confidence(a, b)
                            if confidence >= self.min_confidence:
                                yield ((a, b), confidence)
                                self._verbose(a, b, confidence)

    def _confidence(self, a, b):
        return self.apriori._support(a | b) / self.apriori._support(a)

    def _verbose(self, a, b, c):
        if self.verbose > 0:
            a = ', '.join(sorted(a))
            b = ', '.join(sorted(b))
            print('confidence=%f { %s } -> { %s }' % (c, a, b))

if __name__ == '__main__':
    print('Running mini Apriori test...')

    transactions = [
        {'A', 'C', 'D'},
        {'B', 'C', 'E'},
        {'A', 'B', 'C', 'E'},
        {'B', 'E'}
    ]
    
    print('Transactions:')
    print(transactions)

    apriori = Apriori(transactions, 2, verbose=True)
    rule_iterator = RuleIterator(apriori, min_confidence=0.8, verbose=True)

    for _ in rule_iterator.run_apriori():
        pass

    for _ in rule_iterator.run():
        pass
