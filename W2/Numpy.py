import numpy as np
import utils

baseball = [180, 215, 210, 210, 188, 176, 209, 200]
np_baseball = np.array(baseball)
np_height = np.array([1, 2, 3, 4, 5])
tall = np_height > 3
print(np_height[tall])


baseball_2d = [[180, 78.4],
               [215, 102.7],
               [210, 98.5],
               [188, 75.2]]
np_baseball_2d = np.array(baseball_2d)
print(type(np_baseball_2d))
print(np_baseball_2d.shape)

np_mat = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
print(np_mat * 2)
print(np_mat + np.array([10, 10]))
print(np_mat + np_mat)


positions = ['GK', 'M', 'A', 'D']
heights = [191, 184, 185, 180]
np_positions = np.array(positions)
np_heights = np.array(heights)
gk_heights = np_heights[np_positions == "GK"]
other_heights = np_heights[np_positions != 'GK']
print("Median height of goalkeepers: " + str(np.median(gk_heights)))
print("Median height of other players: " + str(np.median(other_heights)))
