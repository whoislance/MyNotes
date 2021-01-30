import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x +1
y2 = x**2

plt.figure()
plt.plot(x,y1)

plt.figure(num=5,figsize=(6,4))
plt.plot(x,y1)
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')

plt.show()

