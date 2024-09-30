l = [10, 20, 3, 40,50]

l[2] = 30

l = [1, 20, 30, 5 , 6]

l[1:3] = [2, 3, 4]

l[1:3] = 'python'

l = [1, 2, 3, 4, 5, 6, 7, 8]

l[1::2]= 20, 40, 60, 80

del l[2]

l.append(5)

l.extend('python')

l.insert(2,'a')

print(l)

from timeit import timeit

l = []
timeit('l.append(1)',globals=globals(), number=100_000)

