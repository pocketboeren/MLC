import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

x = [1, 2, 3, 4, 5]
y = [23, 25, 23, 48, 33]
plt.plot(x, y)
plt.scatter(x, y)
plt.xscale("log")

c = [1, 1, 2, 4, 5, 6, 3, 7, 6, 8, 1, 3, 7, 4]
plt.hist(c, bins=5)
x_lab = 'X label'
y_lab = 'Y label'
title = 'This is the title of my chart'

# Add axis labels
plt.xlabel(x_lab)
plt.ylabel(y_lab)
plt.title(title)
plt.yticks([0, 1, 2], ["one", "two", "three"])


plt.scatter(x=x,
            y=y,
            s=np.array(x)*50,
            alpha=0.8)
plt.grid(True)

legend = {
    'Asia': 'red',
    'Europe': 'green',
    'Africa': 'blue',
    'Americas': 'yellow',
    'Oceania': 'black'
}
legend_handles = [Line2D([0], [0], linestyle="none", marker="o", alpha=0.6,
                         markersize=12, markerfacecolor=color)
                  for color in legend.values()]

plt.legend(legend_handles, legend.keys(), numpoints=1, fontsize=12, loc="best")
plt.text(3, 35, 'India')
