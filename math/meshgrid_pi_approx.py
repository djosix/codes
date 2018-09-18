import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=E1101

meshgrid = lambda steps: np.meshgrid(np.linspace(0, 1, steps), np.linspace(0, 1, steps))

pis = []

for steps in np.arange(1000) + 1:
    m_x, m_y = meshgrid(steps)
    r2 = m_x ** 2 + m_y ** 2
    pi = 4 * (r2 < 1).mean()
    print('steps: {}, estimated pi ='.format(steps), pi)
    pis.append(pi)

plt.plot(pis)
plt.show()

