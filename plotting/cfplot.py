import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

dfs = []
for i in range(35):
    df = pd.read_csv('output/scores/ideal_kmeans4/stoch_ideal_kmeans_{}.csv'
                     .format(i)) 
    df['counter'] = i
    dfs.append(df)

print('Start merging...')
the_df = dfs[0].append(dfs[1:], ignore_index=False)
the_df['the_index'] = the_df.index


sns.tsplot(time="the_index", value="EV",
           unit="counter", data=the_df)
plt.show()
plt.close()
