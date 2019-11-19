import matplotlib.pyplot as plt
import numpy as np

X = np.random.normal(0,1,1024)
Y = np.random.normal(0,1,1024)
T = np.arctan2(Y,X) # for color value

plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.xlim((-10.5,10.5))
plt.ylim((-10.5,10.5))

plt.show()
