import pandas as pd
import matplotlib.pyplot as plt

import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
dt = float(sys.argv[1])
xlabel = str(sys.argv[2])
legend = str(sys.argv[3])

df = pd.read_csv('tmp.txt', header=None)
df.index += 1
df.index *= dt*20

ax = df.plot()
ax.set_xlabel(xlabel)
ax.legend([legend])

plt.savefig('conv.png')

