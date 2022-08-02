import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.style.use('./CCGAN-seaborn.mplstyle')
plt.switch_backend('agg')


fig = plt.figure()
ax = plt.subplot(111)

num = 20

xax = np.linspace(0.0, 1.0, num + 1, endpoint=True)
yax = np.linspace(0.0, 1.0, num + 1, endpoint=True)

xv, yv = np.meshgrid(xax, yax)

ax.scatter(xv, yv, c='red', edgecolor='none', alpha=0.5, s=25)
ax.set_xlabel("X", fontsize=20)
ax.set_ylabel("Y", fontsize=20)
ax.set_title("Known Distributions", fontsize=25)
ax.set_xmargin(0.05)
ax.set_ymargin(0.05)
ax.grid(visible=True)
plt.savefig("grid.jpg")