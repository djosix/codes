import numpy as np

probs = np.array([
    [0.4, 0.1], # P(A->A) P(B->A)
    [0.6, 0.9]  # P(A->B) P(B->B)
])

transit = lambda x: probs @ x

# Random init A, B
x = np.random.random([2, 1])
print('x:', x.ravel())

# Run until stable
while True:
    x_old = x
    x = transit(x)
    if np.all(x_old == x):
        break

x = x.ravel()
print('x_final:', x)
print('a_b_ratio:', x[0] / x[1])
