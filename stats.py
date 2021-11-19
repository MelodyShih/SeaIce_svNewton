import pandas as pd
import matplotlib.pyplot as plt

import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
dt = float(sys.argv[1])

df = pd.read_csv('stats.out', header=None, delimiter=" ")
df.columns = ["energy", "meandiv", "meanshear", "meanspeed", "meandeform"]
#df.index += 1
df.index *= dt
df.index.name="day"
df["energy"]/=(512*512)
df["meanspeed"]/=(512*512)
df["meandeform"]/=(512*512)
df["meanshear"]/=(512*512)
df["meandiv"]/=(512*512)
print(df)
df.to_csv("1km_stats.csv", float_format="%.5e")

#if multiplot:
#    df2 = pd.read_csv('tmp2.txt', header=None)
#    df2.index += 1
#    df2.index *= dt
#
#    df3 = pd.concat([df, df2], axis=1)
#    df3.columns = ["newton", "stressvel"]
#    ax = df3.plot(style='-o', logy=True, markersize=4)
#    ax.set_xlabel(xlabel)
#    ax.legend(["stdnewton", "stressvel"])
#    ax.set_title("mesh size 4km; delta_min = 2e-10", fontsize=12)
#    ax.set_ylim([2,300]);
#else:
#    df3 = pd.concat([df, df2], axis=1)
#    ax = df3.plot(style='-o', logy=True, markersize=4)
#    ax.set_xlabel(xlabel)
#    ax.legend(legend)
#
#plt.savefig('conv.png')
#plt.show()
