#!/usr/bin/env python3

from utils import *
from hopfield import Hopfield

def test(train, test, title):
    hf = Hopfield()
    hf.fit(train.data)
    result = hf.predict(test.data)
    plot_all(train.data, test.data, result, train.shape, title=title)


try:
    basic, basic_test = Data.basic()
    test(basic, basic_test, 'Basic')

    bonus, bonus_test = Data.bonus()
    test(bonus, bonus_test, 'Bonus')

    for p in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        test(basic, basic.noise(p), 'Basic with noise {}'.format(p))

    for p in [0, 0.01, 0.05, 0.1, 0.2]:
        test(bonus, bonus.noise(p), 'Bonus with noise {}'.format(p))

except KeyboardInterrupt:
    pass
