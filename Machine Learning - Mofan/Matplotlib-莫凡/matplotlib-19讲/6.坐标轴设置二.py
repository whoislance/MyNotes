import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x +1
y2 = x**2


plt.figure(num=5,figsize=(6,4))
plt.plot(x,y1)
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')

plt.xlim((-1,2))
plt.ylim((-2,3))

plt.xlabel('this is x')
plt.ylabel('this is y')

new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.6,-1,1.22,3],
			['terrible','bad','normal',r'$beta$',r'$\alpha$'])

#gca = 'get curretn axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',-1))
ax.spines['left'].set_position(('data',0))

plt.show()

