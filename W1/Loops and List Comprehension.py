i = 0
while i < 10:
    print(i, end=' ')
    i += 1

squares = []
for n in range(10):
    squares.append(n**2)
print(squares)


def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])


print(count_negatives([0, -1, -2, 3, -9]))
