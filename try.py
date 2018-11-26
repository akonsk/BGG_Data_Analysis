import matplotlib.pyplot as plt
import numpy as np
import mplcursors

np.random.seed(42)

labels=['a','b','c','d','e']
fig, ax = plt.subplots()
plt.scatter(*np.random.random((2, 5)))

mplcursors.cursor(ax,hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(labels[sel.target.index])
)

plt.show()

