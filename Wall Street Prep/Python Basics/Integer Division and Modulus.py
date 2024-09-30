a = 10
b = 3

a / b

a // b

-12 / 5

-12 // 5

10 % 3

4 % 2

5 % 2

13567 % 2

165 / 60

165 - (2*60)

165 // 60

165 - (165 // 60 * 60)

165 % 60

elasped_minutes = 165
hours = elasped_minutes // 60
remaining_minutes = elasped_minutes % 60
print(hours, remaining_minutes)

total = 0 
for i in range(1, 1001):
    total += i
    if i % 100 == 0:
        print(f'total = {total}...')
print(f'Final total = {total}')