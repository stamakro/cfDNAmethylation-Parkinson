import pandas as pd
import numpy as np
from rutils import *
import argparse
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, linregress
from statsmodels.stats.multitest import multipletests
import pickle
import sys
from tqdm import tqdm
import matplotlib
matplotlib.style.use('seaborn-paper')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.mixture import GaussianMixture


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clinical', dest='clinical_data', default='')
    parser.add_argument('--dmrs', dest='dmrs', default='../results/selected_features_with_regels_genebody.csv')

    parser.add_argument('--demographics', dest='clinical_data', default='../data/demographics.pkl')
    # change this between the cpg islands (cpgi) and cell type-specific hypermethylated regions (hyper)
    parser.add_argument('--cfdna', dest='data', default='../data/methylation_cpgi.pkl')
    parser.add_argument('--sample-info', dest='info', default='../data/info.pkl')
    parser.add_argument('--external-controls', dest='externalControls', default='../data/external_controls.pkl')


    args = parser.parse_args()

    # loading data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    with open(args.info, 'rb') as f:
        info = pickle.load(f)

    with open(args.clinical_data, 'rb') as f:
        demographics = pickle.load(f)

    # remove low quality samples and patients with uncertain PD diagnosis
    info = info[info['QC_pass']==1]

    data = data.loc[info.index]

    # follow-up samples in separate df
    datav3TPM = data.loc[info[info['timepoint'] == 'FU'].index]

    datav1TPM = data.loc[info[info['timepoint'] == 'BL'].index]
    dataCtrl = data.loc[info[info['timepoint'].isna()].index]

    dmrs = pd.read_csv(args.dmrs, index_col=0)

    sumsV1 = datav1TPM[dmrs.index].sum(axis=1)
    sumsV3 = datav3TPM[dmrs.index].sum(axis=1)



    MIN = 0.9 * np.minimum(sumsV1.min(), sumsV3.min())
    MAX = 1.1 * np.maximum(sumsV1.max(), sumsV3.max())

    fig,ax = plt.subplots(1,2)
    xx = np.linspace(MIN,MAX, 5)
    ax[0].scatter(sumsV1, sumsV3)
    ax[0].plot(xx,xx,'k--')

    # ax[0].axhline(finalMu, color='b', alpha=0.4)
    # ax[0].axvline(finalMu, color='b', alpha=0.4)

    ax[0].set_xlabel('DMR normalized read count v1')
    ax[0].set_ylabel('DMR normalized read count v3')

    pval = wilcoxon(sumsV1, sumsV3)[1]

    ax[1].boxplot([sumsV1, sumsV3], widths=5, showfliers=False, positions=[0, clinical['dt'].median()])

    slopesHyper = []
    for j in range(sumsV1.shape[0]):
        ax[1].scatter([0,clinical.iloc[j]['dt']], [sumsV1[j], sumsV3[j]], color='C0', alpha=0.2)

        [a,b,_,_,_] = linregress([0,clinical.iloc[j]['dt']], [sumsV1[j], sumsV3[j]])
        slopesHyper.append(a)
        xx = np.linspace(0,clinical.iloc[j]['dt'],5)
        ax[1].plot(xx, a*xx + b, color='k', alpha=0.1)

        ax[1].set_title('p=%.5f' % pval)

    # ax[0,1].set_xticklabels(['visit1', 'visit3'])
    ax[1].set_xticks(np.arange(0,48,6))
    ax[1].set_xticklabels(np.arange(0,48,6))
    ax[1].set_ylabel('DMR normalized read count')
    ax[1].set_xlabel('time (months)')



with open(args.externalControls, 'rb') as f:
    hbdtpm = pickle.load(f)

sumsH = dataCtrl[dmrs.index].sum(axis=1)
sumsHext = hbdtpm[dmrs.index].sum(axis=1)

fig,ax = plt.subplots(1,1)

colors = ['C0', 'C0', 'C1', 'C1']
box = ax.boxplot([sumsV1, sumsV3, sumsH, sumsHext], patch_artist=True, medianprops={'color':'k'})

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


for v1, v3 in zip(sumsV1, sumsV3):
    ax.scatter(1, v1, color='#888888')
    ax.scatter(2, v3, color='#888888')

    ax.plot([1,2], [v1, v3], color='k', alpha=0.1)

ax.set_xticklabels(['BL', '2-year follow-up', 'external controls'])

ax.set_ylabel('DMR sum score')
