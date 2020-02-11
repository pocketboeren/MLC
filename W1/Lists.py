this_is_my_first_list = ['This', 'is', 'a', 'list']
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]

# List index starts from 0, so hand[0][0] = J

a = 1
b = 0
a, b = b, a
print(a, b)

