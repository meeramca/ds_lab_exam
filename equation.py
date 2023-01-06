import matplotlib.pyplot as plt
import numpy as np
A = np.array([[1,2],[3,4]])
print("A=\n",A)
b = np.array([[2,4],[6,8]])
print("b=\n", b)
c = np.linalg.inv(A)
X = np.linalg.solve(c,b)
print("Value Of X is :\n",X)


