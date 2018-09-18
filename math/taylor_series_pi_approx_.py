import matplotlib.pyplot as plt

def pi_series():
    s, n = 1, 1
    while True:
        yield s / n
        s, n = -s, n + 2

pis = []
pi = 0

for i, n in enumerate(pi_series()):
    pi += n * 4
    pis.append(pi)
    if i > 9999:
        break

print('pi =', pi)
plt.plot(pis)
plt.show()
