a = 10
b = 10

a == b

c = 10.0

a == c

a is c

a = 10
b = 10.0

a == b

a is b

id(a), id(b)

a is b

10 != 12

10.5 != 10.5

10>= 5

10.5 , 100

a = 1 + 1j
b = 1 + 1j
c = 2 + 2j

a == b
a is b, id(a), id(b)

a < c

0.1 * 3 == 0.3

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self):
        return f'Vector({self.x},{self.y})'

v1 = Vector(1,1)
v2 = Vector(1,1)
v3 = Vector(2,3)