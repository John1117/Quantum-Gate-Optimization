import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...

plt.ion()
fig = plt.figure()

ax = fig.add_subplot(111)

line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
pc = ax.fill_between(x, [-1]*len(x), [1]*len(x), color=(0, 0, 1, 0.2))

for phase in np.linspace(0, 10*np.pi, 50):
    y = np.sin(x + phase)
    line1.set_ydata(y)
    pc.remove()
    pc = ax.fill_between(x, y-1, y+1, color=(0, 0, 1, 0.2))

    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
