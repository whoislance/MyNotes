# export PATH=/home/yule/anaconda3/bin:$PATH

import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
z=np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(x=0.0,ls='dotted',color='k')
plt.axhline(y=0.0,ls='dotted',color='k')
plt.axhline(y=0.5,ls='dotted',color='k')
plt.axhline(y=1.0,ls='dotted',color='k')
plt.show()