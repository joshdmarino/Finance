l1 = [1, 2, 3]

l2 = l1[:]

l1 is l2

l2.append(10)

l3 = l1.copy

l3 is l1

#Above is shallow copies

m1 = [[1, 0, 0],[0, 1, 0][0, 0, 1]]

m2 = m1.copy

m1 is m2

from copy import deepcopy

m2 = deepcopy(m1)

