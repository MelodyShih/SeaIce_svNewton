import pandas as pd
import matplotlib.pyplot as plt

import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
dt = float(sys.argv[1])
xlabel = str(sys.argv[2])
legend = str(sys.argv[3])
multiplot = bool(sys.argv[4])

df = pd.read_csv('tmp.txt', header=None)
df.index += 1
df.index *= dt

if multiplot:
	df2 = pd.read_csv('tmp2.txt', header=None)
	df2.index += 1
	df2.index *= dt

	df3 = pd.concat([df, df2], axis=1)
	df3.columns = ["newton", "stressvel"]
	df3.index.name="day"
	df3.to_csv("2km.csv", na_rep=' ')
	ax = df3.plot(style='-o', logy=True, markersize=4)
	ax.set_xlabel(xlabel)
	ax.legend(["stdnewton", "stressvel"])
	#ax.set_title("mesh size 4km; delta_min = 2e-10", fontsize=12)
	ax.set_ylim([2,500]);
else:
	df3 = pd.concat([df, df2], axis=1)
	ax = df3.plot(style='-o', logy=True, markersize=4)
	ax.set_xlabel(xlabel)
	ax.legend(legend)

plt.savefig('conv.png')
plt.show()

