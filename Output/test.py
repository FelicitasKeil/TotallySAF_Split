import numpy as np
from os import sys

A = np.loadtxt("Fisher_GCph_XSAF")
A = A[7:17,7:17]

zp = np.array([0.209366, 0.488624, 0.618284, 0.732253, 0.84311, 0.958236, 1.08543, 1.23761 ,1.44802 ,2.65249])
bp = np.sqrt(1.+zp)

CA = np.sqrt(np.diag(np.linalg.inv(A)))
MA = CA/bp

B = np.loadtxt("Fisher_GCs_SpecSAF")
B = B[7:12,7:12]

bs = np.array([1.4541449979, 1.5842634192, 1.7113829800, 1.8258387238, 1.9355336670])

CB = np.sqrt(np.diag(np.linalg.inv(B)))
print(CB)
sys.exit()
MB = CB/bs

print(np.sqrt(MA[5:]**2+MB**2))
print("")

A = np.insert(A, [10,10,10,10,10], 0, axis=0)
A = np.insert(A, [10,10,10,10,10], 0, axis=1)

B = np.insert(B, [0,0,0,0,0,0,0,0,0,0], 0, axis=0)
B = np.insert(B, [0,0,0,0,0,0,0,0,0,0], 0, axis=1)

C = A+B
CC = np.sqrt(np.diag(np.linalg.inv(C)))

print(CC)