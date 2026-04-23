import matplotlib
matplotlib.style.use('seaborn-paper')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support, roc_curve
import numpy as np


files = ['demographic_alone.pkl',  'demographic.pkl',  'hyper.pkl', 'medseq.pkl']
names = ['demographic', 'demographic+cfDNA concentration', 'cfDNA (cell type markers)', 'cfDNA (CpG islands)']
colors = ['C0', 'C3', 'C5', 'C9']

figROC, axROC = plt.subplots(1,1)
figPR, axPR = plt.subplots(1,1)

nFolds = 3

for file, name, col in zip(files, names, colors):
    with open('../results/posteriors/%s' % file, 'rb') as f:
        dd = pickle.load(f)

    posteriorArchive = dd['post']
    labelArchive = dd['y']
    allthresholds = np.sort(dd['thr'])


    precision = np.zeros((nFolds, allthresholds.shape[0]))
    # also sensitivity
    recall = np.zeros(precision.shape)
    specificity = np.zeros(precision.shape)

    linestyles = ['-', ':', '-.']

    for i in range(nFolds):
        post = posteriorArchive[i]
        ytest = labelArchive[i]

        for j, t in enumerate(allthresholds):
            pred = (post >= t).astype(int)
            pr, rc, _, _ = precision_recall_fscore_support(ytest, pred, average=None)

            if np.sum(pred) > 0:
                precision[i,j] = pr[1]
            else:
                precision[i,j] = np.nan

            recall[i,j] = rc[1]
            specificity[i,j] = rc[0]


    muPrecision = np.nanmean(precision,0)
    stderrPrecision = np.nanstd(precision,axis=0, ddof=1) / np.sqrt(nFolds)
    muFPR = 1-np.nanmean(specificity,0)
    muRecall = np.nanmean(recall,0)
    stderrRecall = np.nanstd(recall,axis=0, ddof=1) / np.sqrt(nFolds)

    axROC.plot(muFPR, muRecall, color=col, label=name)
    axROC.fill_between(muFPR, muRecall - stderrRecall, muRecall + stderrRecall, color=col, alpha=0.1)

    axPR.plot(muRecall, muPrecision, color=col, label=name)
    axPR.fill_between(muRecall, muPrecision - stderrPrecision, muPrecision + stderrPrecision, color=col, alpha=0.1)



xx = np.linspace(0,1,5)
axROC.plot(xx,xx,'k--')
axROC.legend()
axROC.set_xlim(-0.02,1.02)
axROC.set_ylim(-0.02,1.02)
axROC.set_xlabel('False positive rate', fontsize=13)
axROC.set_ylabel('True positive rate', fontsize=13)
plt.setp(axROC.get_xticklabels(), fontsize=12)
plt.setp(axROC.get_yticklabels(), fontsize=12)
plt.tight_layout()

axPR.axhline(np.array(dd['y']).flatten().mean(), color='k', linestyle='--')
axPR.legend()
axPR.set_xlim(-0.02,1.02)
axPR.set_ylim(-0.02,1.02)
axPR.set_xlabel('Recall', fontsize=13)
axPR.set_ylabel('Precision', fontsize=14)
plt.setp(axPR.get_xticklabels(), fontsize=12)
plt.setp(axPR.get_yticklabels(), fontsize=12)
plt.tight_layout()

figROC.savefig('../figures/roc_curves.png', dpi=600)
figPR.savefig('../figures/pr_curves.png', dpi=600)

plt.show()
