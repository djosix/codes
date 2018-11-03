from apriori import Apriori, RuleIterator
import pickle, os


min_support = 600
min_confidence = .6


with open('./online_retail.pkl', 'rb') as f:
    data = pickle.load(f)
    transactions = list(data['transactions'].values())


apriori = Apriori(transactions, min_support, True)
rule_iterator = RuleIterator(apriori, min_confidence, False)

for _ in rule_iterator.run_apriori():
    pass

for rule, confidence in rule_iterator.run():
    a, b = rule
    s1 = [data['descriptions'][code] for code in sorted(a)]
    s2 = [data['descriptions'][code] for code in sorted(b)]
    print('%.2f: %s -> %s' % (confidence, s1, s2))
