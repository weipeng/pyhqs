import pandas as pd
from hierarchies.hierarchy import Hierarchy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, gridspec
from datahandler import (load_mat_data, load_cluster)
import seaborn as sns

sns.set_context('talk')
sns.set_style('darkgrid')

dfs = []

method_short_names = {
    'kmeans3': 'HKM3', 'kmeans4': 'HKM4', 'kmeans5': 'HKM5', 
    'agg': 'AC', 'randagg': 'RAC'
}

for i, method in enumerate(['agg', 'kmeans3', 'kmeans4']):
    gs1 = gridspec.GridSpec(1, 4)
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.4))
    #fig = plt.figure(figsize=(18, 6))
    for j, data_name in enumerate(['r15', 'compound', 'toy_data']):
        #plt.subplot(1, 3, j+1)
        dfs = []
        tmp_method = 'kmeans' if 'kmeans' in method else method
        scores3 = pd.read_csv('output/scores/test/{}_{}/stoch_{}_{}_0.csv'
                              .format(data_name, method, data_name, method))
        for k in range(1, 50):
            df = pd.read_csv('output/scores/test/{}_{}/stoch_{}_{}_{}.csv'
                             .format(data_name, method, data_name, method, k))
            scores3 += df
        scores3 /= 50

        data3 = pd.read_csv('./data/{}/{}.txt'.format(data_name, data_name),
                            sep=' +|\t|,', engine='python', header=None)
        show = axes[j].scatter(data3.values[:, 0], data3.values[:, 1],
                               c=scores3.EV.tolist(), edgecolor='k', alpha=.725,
                               vmin=-1, vmax=1, cmap="RdYlBu", s=43)
        #plt.colorbar()
        mthd_in_short = method_short_names[method]
        axes[j].set_title('{} on {}'.format(mthd_in_short, data_name.capitalize()))
    #    plt.savefig('./new_figures/{}_{}_spectrum.eps'.format(data_name, method), dpi=1600)

    
    #cbaxes = fig.add_axes([.91, .3, .03, .9]) 
    cax = fig.add_axes([.92, 0.07, 0.011, 0.83])
    fig.colorbar(show, cax=cax)
    fig.subplots_adjust(hspace=0.5, left=0.035)

    #plt.tight_layout()
    plt.savefig('{}_spectrum.pdf'.format(method), dpi=900)
    plt.close('all')


