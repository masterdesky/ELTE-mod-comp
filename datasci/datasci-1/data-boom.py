import numpy as np
import matplotlib.pyplot as plt

xmin: int = 2010
xmax: int = 2035
x = np.linspace(xmin, xmax, xmax-xmin+1)
y = np.exp(range(0, xmax-xmin+1))


fig, ax = plt.subplots(figsize=(18, 5))

color = 'tab:blue'
alpha = 0.7
ax.plot(x, y,
        color=color, alpha=alpha)
ax.fill_between(x=x, y1=0, y2=y,
                color=color, alpha=alpha)

plt.savefig('presentation/src/img/data-boom.png',
            bbox_inches='tight')
plt.show()
