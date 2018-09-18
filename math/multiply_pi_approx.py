import matplotlib.pyplot as plt

max_iter = 200

pis = []

pi_2 = 1
num = 2
den = 1
for i in range(max_iter):
    pi_2 *= num / den
    pi = pi_2 * 2
    
    print(pi)
    pis.append(pi)

    if i % 2 == 0:
        den += 2
    else:
        num += 2

plt.plot(pis)
plt.show()
