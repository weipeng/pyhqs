import os

import sys

import pandas as pd
from hierarchies.hierarchy import Hierarchy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from datahandler import (load_mat_data, load_cluster)


# READ DATA
data = pd.read_csv('./data/toy_data/toy_data.txt', sep=' +|\t|,', engine='python', header=None)
X = data.values

# colors = np.random.rand(X.shape[0])

plt.scatter(X[:,0], X[:,1], alpha=0.5)
plt.show()
plt.title('Ideal User Case : 16 clusters')
plt.grid()



data2 = pd.read_csv('./data/compound/compound.txt', sep=' +|\t', engine='python')
X2 = data.values[:, :2]

colors = [int(i % 23) for i in data2[2].values]
plt.scatter(X2[:,0], X2[:,1], c=colors, cmap='jet',alpha=0.5)
plt.show()
plt.title('Compound dataset')
plt.grid()


data3 = pd.read_csv('./data/r15/r15.txt', sep=' +|\t', engine='python', header=None)
X3 = data3.values[:, :2]

colors = [int(i % 23) for i in data3[2].values]
plt.scatter(X3[:,0], X3[:,1], c=colors, cmap='jet',alpha=0.5)
plt.show()
plt.title('R15 dataset')
plt.grid()




# Ideal

sidealhk3 = pd.read_csv('./output/scores/test/toy_data/stoch_toy_data_kmeans_3.csv', sep=',', engine='python')
sidealhk4 = pd.read_csv('./output/scores/test/toy_data/stoch_toy_data_kmeans_4.csv', sep=',', engine='python')
sidealhk5 = pd.read_csv('./output/scores/test/toy_data/stoch_toy_data_kmeans_5.csv', sep=',', engine='python')
sidealrand = pd.read_csv('./output/scores/test/toy_data/stoch_toy_data_randagg.csv', sep=',', engine='python')
sidealagg = pd.read_csv('./output/scores/test/toy_data/stoch_toy_data_agg.csv', sep=',', engine='python')

plt.figure()
plt.plot(np.sort(sidealhk3.EV))
plt.plot(np.sort(sidealhk4.EV))
plt.plot(np.sort(sidealhk5.EV))
plt.plot(np.sort(sidealrand.EV))
plt.plot(np.sort(sidealagg.EV))
plt.title('Evaluation - Ideal dataset')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.grid()
plt.legend(['hk3','hk4','hk5','rand','agg'])
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/Score_IdealData.eps', format='eps', dpi=1000)

# Compound
scomphk3 = pd.read_csv('./output/scores/stoch_compound_hkmeans_3.csv', sep=',', engine='python')
scomphk4 = pd.read_csv('./output/scores/stoch_compound_hkmeans_3.csv', sep=',', engine='python')
scomphk5 = pd.read_csv('./output/scores/stoch_compound_hkmeans_3.csv', sep=',', engine='python')
scomprand = pd.read_csv('./output/scores/stoch_compound_randAgg.csv', sep=',', engine='python')
scompagg = pd.read_csv('./output/scores/stoch_compound_AGG.csv', sep=',', engine='python')

plt.figure()
plt.plot(np.sort(scomphk3.EV))
plt.plot(np.sort(scomphk4.EV))
plt.plot(np.sort(scomphk5.EV))
plt.plot(np.sort(scomprand.EV))
plt.plot(np.sort(scompagg.EV))
plt.title('Evaluation - Compound dataset')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.grid()
plt.legend(['hk3','hk4','hk5','rand','agg'])
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/Score_CompoundData.eps', format='eps', dpi=1000)


# LEVEL 1-2 HKMEANS_3
del X2
X2 = csr_matrix(data2.values[:, :2])
tree, Z = load_cluster('./output', 'compound_hkmeans_3')
hier = Hierarchy(X2,Z)
label = np.zeros([X2.shape[0],1])
i = 1
for el in Z[-1]:
    idx = hier.get_items_at_node(el)
    label[idx] = i
    i += 1
colors2 = [int(i % 23) for i in label]
plt.scatter(data2.values[:,0], data2.values[:,1], c=colors2, cmap='jet',alpha=0.5)
plt.show()
plt.title('Compound hkmeans 3 - 1st level')
plt.grid()
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/Compound_hkmeans3_l1.eps', format='eps', dpi=1000)



label = np.zeros([X2.shape[0],1])
i = 1
for el in Z[-1]:
    el= el - X2.shape[0]
    for el2 in Z[el]:
        idx = hier.get_items_at_node(el2)
        label[idx] = i
        i += 1
colors2 = [int(i % 23) for i in label]
plt.scatter(data2.values[:,0], data2.values[:,1], c=colors2, cmap='jet')
plt.show()
plt.title('Compound hkmeans 3 - 2st level')
plt.grid()
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/Compound_hkmeans3_l2.eps', format='eps', dpi=1000)




# R15


#
sr15hk3 = pd.read_csv('./output/scores/stoch_r15_hkmeans_3.csv', sep=',', engine='python')
sr15hk4 = pd.read_csv('./output/scores/stoch_r15_hkmeans_3.csv', sep=',', engine='python')
sr15hk5 = pd.read_csv('./output/scores/stoch_r15_hkmeans_3.csv', sep=',', engine='python')
sr15rand = pd.read_csv('./output/scores/stoch_r15_randAgg.csv', sep=',', engine='python')
sr15agg = pd.read_csv('./output/scores/stoch_r15_AGG.csv', sep=',', engine='python')

plt.figure()
plt.plot(np.sort(sr15hk3.EV))
plt.plot(np.sort(sr15hk4.EV))
plt.plot(np.sort(sr15hk5.EV))
plt.plot(np.sort(sr15rand.EV))
plt.plot(np.sort(sr15agg.EV))
plt.title('Evaluation - R15 dataset')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.grid()
plt.legend(['hk3','hk4','hk5','rand','agg'])
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/Score_r15Data.eps', format='eps', dpi=1000)


# LEVEL 1-2 HKMEANS_3
del X3
X3 = csr_matrix(data3.values[:, :2])
tree, Z = load_cluster('./output', 'r15_hkmeans_3')
hier = Hierarchy(X3,Z)
label = np.zeros([X3.shape[0],1])
i = 1
for el in Z[-1]:
    idx = hier.get_items_at_node(el)
    label[idx] = i
    i += 1
colors3 = [int(i % 23) for i in label]
plt.scatter(data3.values[:,0], data3.values[:,1], c=colors3, cmap='jet',alpha=0.5)
plt.show()
plt.title('r15 hkmeans 3 - 1st level')
plt.grid()
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/r15_hkmeans3_l1.eps', format='eps', dpi=1000)



label = np.zeros([X3.shape[0],1])
i = 1
for el in Z[-1]:
    el= el - X3.shape[0]
    for el2 in Z[el]:
        idx = hier.get_items_at_node(el2)
        label[idx] = i
        i += 1
colors3 = [int(i % 23) for i in label]
plt.scatter(data3.values[:,0], data3.values[:,1], c=colors3, cmap='jet')
plt.show()
plt.title('r15 hkmeans 3 - 2st level')
plt.grid()
plt.savefig('/Users/raulms/GitRep/ModelingTransitions/Figures/r15_hkmeans3_l2.eps', format='eps', dpi=1000)
