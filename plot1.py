import numpy as np
import matplotlib.pyplot as plt

g= 9.81
v0 = 20
h0 = 50

t = np.linspace(0, 6, 200)
h  = h0 + v0*t - 0.5*g*t**2

plt.plot(t, h)
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Projectile Motion")
plt.grid(True)
plt.show()