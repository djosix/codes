from decision_tree import DecisionTree, print_tree
from data import Dataset

dataset = Dataset()
dataset.split([1, 0])

clf = DecisionTree(dataset.F, dataset.C, splits=20)
clf.fit(*dataset.train)

print_tree(clf._tree)
