import matplotlib.pyplot as plt
import numpy as np


def v(d, a):
    return np.sqrt((d * 9.81) / np.sin(2 * np.radians(a)))


x_distance = np.arange(0, 100, 1)
x_angle = np.arange(1, 90, 1)

X, Y = np.meshgrid(x_distance, x_angle)
Z = v(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
ax.set_xlabel('dystans')
ax.set_ylabel('kat')
ax.set_zlabel('moc rzutu')
plt.show()
