import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist

x = np.arange(-1, 10, 0.1)

y1 = -2 * x + 5
y2 = -x + 3

px = [3, 4]
py = [3, 3]
nx = [1]
ny = [1]

fig = plt.figure(figsize=(8, 8))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)

ax.axis[:].set_visible(False)
ax.axis['x'] = ax.new_floating_axis(0, 0)
ax.axis['x'].set_axisline_style('->', size=1.0)
ax.axis['y'] = ax.new_floating_axis(1, 0)
ax.axis['y'].set_axisline_style('-|>', size=1.0)
ax.axis['x'].set_axis_direction('top')
ax.axis['y'].set_axis_direction('right')

plt.xlim(-1, 6)
plt.ylim(-1, 6)

plt.scatter(px, py, marker='o', color='green')
plt.scatter(nx, ny, marker='x', color='red')

plt.annotate('x1', (3, 3.1))
plt.annotate('x2', (4, 3.1))
plt.annotate('x3', (1.1, 1.1))
plt.annotate('x(1)', (5.9, -0.2))
plt.annotate('x(2)', (-0.4, 6))
plt.annotate('2x(1)+x(2)-5=0', (0.5, 4.7))
plt.annotate('x(1)+x(2)-3=0', (3, 0.5))

plt.plot(x, y1, 'b')
plt.plot(x, y2, 'b')

plt.show()
