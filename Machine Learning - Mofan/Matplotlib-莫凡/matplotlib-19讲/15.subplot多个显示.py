import matplotlib.pyplot as plt
import numpy as np

plt.figure()

plt.subplot(2,2,1)
plt.plot([0,1],[0,1])

plt.subplot(2,2,2)
plt.plot([0,1],[0,2])

plt.subplot(223)
plt.plot([0,1],[0,3])

plt.subplot(224)
plt.plot([0,1],[0,4])

plt.show()