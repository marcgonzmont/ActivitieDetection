import numpy as np

x = np.empty((0,1), int)
y = [[1] [2]]
z = np.concatenate([x,y])
x = [[1] [2] [3]]
z = np.concatenate([z,x])
print(z)