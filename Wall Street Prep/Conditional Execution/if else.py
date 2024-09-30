if 1 < 2:
    print('1 is less than 2')

if 1 < 2:
    print('block - line 1')
    print('block - line 2')
    print('block - line 3')
print('next line')

if 1 < 2:
    print('1 is less than 2')
else:
    print('1 is not less than 2')

account_enabled = True
balance = 1000
withdraw = 100

if account_enabled and withdraw <= balance:
    print('authorized')
else:
    print('not authorized')

grade = 72
letter_grade = 'F'

if grade >= 90:
    letter_grade = 'A'
elif grade >= 80:
    letter_grade = 'B'
elif grade >= 70:
    letter_grade = 'C'
elif grade >= 60:
    letter_grade = 'D'

print(letter_grade)