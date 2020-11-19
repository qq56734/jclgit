import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0, 0.2, 0.05)
# print(x)
# y = np.array([0, 1, 2, 3.5])


x = np.arange(0, 0.08, 0.02)
print(x)
y = np.array([0, 5, 15, 25.7])

z1 = np.polyfit(x, y, 2)#用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) #在屏幕上打印拟合多项式
newx = np.arange(0, 1, 0.0001)
yvals=p1(newx)#也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(newx, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()


print(np.roots([28.57, 16.43, 0.03571 - 6]))