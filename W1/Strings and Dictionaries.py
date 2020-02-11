planet = 'Pluto'
len(planet)
planet.upper()
planet.index('u')
date_str = '1956-01-31'
year, month, day = date_str.split('-')
'/'.join([month, day, year])
position = 9
print(planet + ", you'll always be the " + str(position) + "th planet to me.")
print("{}, you'll always be the {}th planet to me.".format(planet, position))


numbers = {'one':1, 'two':2, 'three':3}
print(numbers['one'])
for k in numbers:
    print("{} = {}".format(k, numbers[k]))

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
print(planet_to_initial)
