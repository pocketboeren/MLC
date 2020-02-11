def least_difference(a, b, c):
    """

    This function returns the smallest difference between all three inputs

    :param a: First value
    :param b: Second value
    :param c: Third value
    :return: Smallest difference
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)


help(least_difference)
print(10, '08', 1991, sep='-')


def greet(who="Colin"):
    print("Hello,", who)


greet()
greet(who="Kaggle")
greet("world")
