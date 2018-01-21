import numpy as np
from itertools import product


x = ['Right']*3
# print(x)
y = ['Left']*5
# print(y)
z = ['Other']*2
# print(z)
comb = [comb for comb in product(x, z)]
final = np.concatenate([x,y,z])
test = np.empty_like(final)
# print(final.size)
# print(final)
# print(test.shape)

for i in range(final.size):
    # test = np.concatenate([test, 'Right'])
    test[i] = 'foo'
    # print(i)
print(test)