1 + 0.5

2 * 1.125

18 / 4

2 ** 8

2 ** (-8)

4 ** 0.5

(-4) ** 0.5

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
v2 = Vector(2,3)

v1 + v2
