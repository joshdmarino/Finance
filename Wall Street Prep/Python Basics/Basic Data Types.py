# Basic Data Types
int = 0, 1, 100, -100
float = 3.14, -1.3
boolean = True, False

# Not exact representation
format(0.1, '.25f')

# Exact represenation
format(0.125, '.25f')

# True, because 1 has an exact representation
1 + 1 + 1 == 3

# True, because 0.125 has an exact representation

0.125 + 0.125 + 0.125 == 0.375

# False, because 0.1 has an exact representation
0.1 + 0.1 + 0.1 == 0.3

# There's a small difference
format(0.1 + 0.1 + 0.1, '.25f')
format(0.3, '.25f')
