import os

import sys

import pandas as pd
from hierarchies.hierarchy import Hierarchy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from datahandler import (load_mat_data, load_cluster)
import seaborn as sns
import random
np.random.seed(77777)

sns.set_style('white')
sns.set('paper')
sns.despine()

markers = ['.', 'o', '>', 'h', '*', 'P', 'v', '<', 's', 'd']
#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(50)]

dn = 'r15'
DN = dn.upper()
data3 = pd.read_csv('./data/{}/{}.txt'.format(dn, dn), sep=' +|\t', 
                    engine='python', header=None)
X3 = csr_matrix(data3.values[:, :2])
tree, Z = load_cluster('./output/r15_agg', 'r15_agg_3')
hier = Hierarchy(X3, Z)
label = np.zeros(X3.shape[0], dtype=int)
i = 1
for el in Z[-1]:
   idx = hier.get_items_at_node(el)
   label[idx] = i
   i += 1
colors3 = [colors[i] for i in label]

fig = plt.figure()
ax = fig.add_subplot(111)
for k in np.unique(label):
    inds = np.where(label == k)
    plt.scatter(data3.values[inds,0], data3.values[inds,1], 
                #c='white', 
                marker=markers[k],
                c=colors[k],    
                markersize=1,
                alpha=.3, edgecolor=colors[k])
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.title('HKM3 on {} - first level'.format(DN), fontsize=24)
#plt.tight_layout()
plt.savefig('./{}_hkmeans3_l1.pdf'.format(DN), dpi=1600)
plt.close()


label = np.zeros(X3.shape[0], dtype=int)
i = 1
for el in Z[-1]:
   el = el - X3.shape[0]
   for el2 in Z[el]:
       idx = hier.get_items_at_node(el2)
       label[idx] = i
       i += 1
colors3 = [colors[i] for i in label]

fig = plt.figure()
ax = fig.add_subplot(111)
for k in np.unique(label):
    inds = np.where(label == k)
    plt.scatter(data3.values[inds,0], data3.values[inds,1],
                c='white', marker=markers[k],
                alpha=.8, edgecolor=colors[k])
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('HKM3 on {} - second level'.format(DN), fontsize=24)
#plt.tight_layout()
plt.savefig('./{}_hkmeans3_l2.eps'.format(DN), dpi=1600)
plt.close()

label = np.zeros(X3.shape[0], dtype=int)
i = 1
for el in Z[-1]:
    el = el - X3.shape[0]
    for el2 in Z[el]:
        el2 = el2 - X3.shape[0]
        for el3 in Z[el2]:
            idx = hier.get_items_at_node(el3)
            label[idx] = i
            i += 1

fig = plt.figure()
ax = fig.add_subplot(111)
for k in np.unique(label):
    inds = np.where(label == k)
    plt.scatter(data3.values[inds, 0], data3.values[inds, 1],
                c='white', marker=markers[k],
                alpha=.8, edgecolor=colors[k])
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('HKM3 on {} - third level'.format(DN), fontsize=24)
#plt.tight_layout()
plt.savefig('./{}_hkmeans3_l3.eps'.format(DN), dpi=1600)
plt.close()
