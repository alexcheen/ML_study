
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

for i, color in enumerate("rgby"):
    plt.subplot(221 + i, facecolor=color)

plt.show()