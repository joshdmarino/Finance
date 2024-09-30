# Objects

class Account:
    def __init__(self, account_number, account_type, initial_balance):
        self.account_number = account_number
        self.account_type = account_type
        self.balance = initial_balance

    def deposit(self, amount):
        # should also check that amount is a numerical value!
        if amount > 0:
            self.balance = self.balance + amount
            print(f'Deposited {amount}')
            print(f'New balance is: {self.balance}')
        else:
            print(f'{amount} is an invalid amount.')

    def withdraw(self, amount):
        # should also check that amount is a numerical value
        if amount > 0 and amount <= self.balance:
            self.balance = self.balance - amount
            print(f'Withdrawl: {amount}')
            print(f'New balance is: {self.balance}')
        else:
            if amount < 0:
                print(f'{amount} is an invalid amount')
            else:
                print('Insufficient funds.')
                print(f'Current balance is {self.balance}')

my_account = Account('123-456', 'savings', 1000.00)

my_account.account_number

my_account.account_type

my_account.balance

my_account.deposit(100)

my_account.withdraw(600)

my_account.withdraw(10000)