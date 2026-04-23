import pandas as pd
import numpy as np
#from rutils import *
import argparse
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pickle
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_score, roc_curve, average_precision_score, precision_recall_curve
import matplotlib
matplotlib.style.use('seaborn-paper')
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import sys
import os

def featureSelection(Xtrain: np.ndarray, ytrain: np.ndarray, Ctrain: pd.DataFrame, absolute=True):
    '''
    Xtrain: N x F feature matrix, np array
    ytrain: N label vector, np array
    Ctrain: N x C matrix with covariates, pd df
    RETURNS:
        t: F vector with absolute t scores (in the original order)
        p: F vector with uncorrected p-values (in the original order)
        ind: F vector with sorted indices of features, best feature at position 0
    '''
    p = np.zeros(Xtrain.shape[1])
    t = np.zeros(Xtrain.shape[1])

    for i in tqdm(range(p.shape[0])):
        # the code relies on methylation being the first column
        df = pd.DataFrame({'methylation': Xtrain[:,i], 'group': ytrain}, index=np.arange(Xtrain.shape[0]))
        Ctrain = pd.DataFrame(data=Ctrain, index=df.index, columns=[('c%d' % n) for n in range(Ctrain.shape[1])])

        df = pd.concat((df, Ctrain),axis=1)
        formulaANCOVA = 'methylation ~ 1 + ' + ' + '.join(df.columns[1:])

        model = ols(data=df, formula=formulaANCOVA)
        res = model.fit()
        p[i] = res.pvalues['group']
        t[i] = res.tvalues['group']

    if absolute:
        ind = np.argsort(np.abs(t))[::-1]
    else:
        ind = np.argsort(t)[::-1]

    return (t, p, ind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--demographics', dest='clinical_data', default='../data/demographics.pkl')
    # change this between the cpg islands (cpgi) and cell type-specific hypermethylated regions (hyper)
    parser.add_argument('--cfdna', dest='data', default='../data/methylation_cpgi.pkl')
    parser.add_argument('--sample-info', dest='info', default='../data/info.pkl')
    parser.add_argument('--auprc', dest='maximizeAUPRC', default=1, type=int)
    # whether to use methylation features(medseq), demographic features (demographics) or the concatenation of the two (both) for classification
    parser.add_argument('--dataset', dest='dataset', default='medseq')
    # if classifying using demographic features, whether to include cfDNA concentration
    parser.add_argument('--concentration', dest='conc', default=1, type=int)
    # if methylation features are used, whether or not to restrict DMRs to hypermethlated in PD
    parser.add_argument('--positive-only', dest='pos', default=0, type=int)
    # path to external control dataset
    parser.add_argument('--external-controls', dest='externalControls', default='../data/methylation_cpgi.pkl')


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
    datav3 = data.loc[info[info['timepoint'] == 'FU'].index]
    # only keep BL and CTRLs in main data frame
    data = data.loc[info[info['timepoint'] != 'FU'].index]

    # class labels, 1: PD, 0: CTRL
    labels = np.array(pd.Series(data.index).apply(lambda x: 0 if 'CTRL' in x  else 1))

    patientIDs = ['_'.join(c.split('_')[:2]) for c in data.index]

    # covariates
    covs = demographics.loc[patientIDs]

    if args.dataset == 'demographic' and args.conc:
        # make prediction with demographic variables + cfdna concentration
        conc = np.zeros(covs.shape[0])
        for i in range(covs.shape[0]):
            if labels[i] == 0:
                conc[i] = info.loc[covs.index[i]]['cfdnaConcentration']

            else:
                conc[i] = info.loc[covs.index[i]+'_BL']['cfdnaConcentration']

        covs['concentration'] = conc

    covs = np.array(covs)

    #
    #
    if args.dataset == 'demographic':
        if not args.conc:
            dataTPM = pd.DataFrame(covs, columns=['age_BL', 'sexMale', 'bmi'])
        else:
            dataTPM = pd.DataFrame(covs, columns=['age_BL', 'sexMale', 'bmi', 'concentration'])
            # need to remove nans
            tokeep = np.where(~dataTPM['conc'].isna())[0]
            dataTPM = dataTPM.iloc[tokeep]
            labels = labels[tokeep]

    elif args.dataset == 'loyfer':
        with open('../data/loyfer-markers-hyper.pkl', 'rb') as f:
            dd = pickle.load(f)
        dataTPM = dd['X']
        labels = dd['y']

    else:
        dataTPM = data

        if args.dataset == 'both':
            dataTPM = np.hstack((dataTPM, covs))

    nperm = 2000
    np.random.seed(42)
    outerCV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 3 folds x 7 metrics
    outerPerformance = np.zeros((3,7))
    outerPerformanceRnd = np.zeros((3,nperm,7))

    # 3 axis, one for each feature set
    fig, ax = plt.subplots(1,2)

    lls = ['-', ':', '-.']

    nNeighbors = [1,3,5,7,9]
    metrics = ['euclidean', 'cosine']
    nFeatures = [5,10,20,50,75,100,125, 150, 200,500,1000, 2000, 5000, 10000, 15000, 27923]
    alphas = [1e-3, 5e-3, 0.01, 0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.,20.,50.,100.,200.]

    estimators = [10, 20, 50, 100, 200]
    criteria = ['gini', 'entropy']
    depths = [1,2,3]
    featureMax = ['sqrt', 'log2']

    allthresholds = np.array([])

    labelArchive = []
    posteriorArchive = []
    predictionArchive = []



    # nested CV for evaluation
    for i, (trainInd, testInd) in enumerate(outerCV.split(dataTPM, labels)):
        Xtrain = dataTPM.iloc[trainInd]
        ytrain = labels[trainInd]

        Xtest = dataTPM.iloc[testInd]
        ytest = labels[testInd]
        labelArchive.append(ytest)

        Ctrain = covs[trainInd]
        Ctest = covs[testInd]

        # 3 folds x all hyperparam combos
        performanceRF = np.zeros((3, len(nFeatures), len(estimators), len(criteria), len(depths), len(featureMax)))
        performanceLR = np.zeros((3, len(nFeatures), len(alphas)))

        innerCV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for j, (trainIndInner, testIndInner) in enumerate(innerCV.split(Xtrain, ytrain)):
            print('\n\n%d, %d' % (i,j))
            XtrainInner = Xtrain.iloc[trainIndInner]
            XtestInner = Xtrain.iloc[testIndInner]

            ytrainInner = ytrain[trainIndInner]
            ytestInner = ytrain[testIndInner]

            CtrainInner = Ctrain[trainIndInner]
            CtestInner = Ctrain[testIndInner]

            if args.dataset == 'medseq':
                tt, pp, featureInd = featureSelection(np.array(XtrainInner), ytrainInner, CtrainInner[:,:2], absolute=1-args.pos)
                if args.pos:
                    featureInd =  np.array([fi for fi in featureInd if tt[fi]>0])

            elif args.dataset == 'demographic':
                featureInd = np.arange(Xtrain.shape[1])

            else:
                assert args.dataset == 'both'
                featureInd = featureSelection(np.array(XtrainInner), ytrainInner, np.zeros((XtrainInner.shape[0],0)), absolute=1-args.pos)[2]

            ss = StandardScaler()
            cc = XtrainInner.columns
            XtrainInner = pd.DataFrame(ss.fit_transform(XtrainInner), columns=cc)
            XtestInner = pd.DataFrame(ss.transform(XtestInner), columns=cc)

            for k, nfeat in enumerate(nFeatures):
                XtrainInnerF = XtrainInner.iloc[:, featureInd[:nfeat]]
                XtestInnerF = XtestInner.iloc[:, featureInd[:nfeat]]

                for l, aa in enumerate(alphas):
                    clf = LogisticRegression(penalty='l2', C=aa, random_state=42)
                    clf.fit(XtrainInnerF, ytrainInner)

                    # pred = clf.predict(XtestInnerF)
                    # performance[0,j,k,l,m] = precision_score(ytestInner, pred)
                    pred = clf.predict_proba(XtestInnerF)[:,1]
                    if args.maximizeAUPRC:
                        performanceLR[j,k,l] = average_precision_score(ytestInner, pred)
                    else:
                        performanceLR[j,k,l] = roc_auc_score(ytestInner, pred)


                for l, nest in enumerate(estimators):
                    for m, criterion in enumerate(criteria):
                        for n, dp in enumerate(depths):
                            for o, fm in enumerate(featureMax):
                                clf = RandomForestClassifier(n_estimators=nest, criterion=criterion, max_depth=dp, max_features=fm, random_state=42)
                                clf.fit(XtrainInnerF, ytrainInner)

                                pred = clf.predict_proba(XtestInnerF)[:,1]
                                if args.maximizeAUPRC:
                                    performanceRF[j,k,l,m,n,o] = average_precision_score(ytestInner, pred)
                                else:
                                    performanceRF[j,k,l,m,n,o] = roc_auc_score(ytestInner, pred)



        meanPerformance = np.mean(performanceRF,axis=0)
        meanPerformanceLR = np.mean(performanceLR,axis=0)

        if np.max(meanPerformance) > np.max(meanPerformanceLR):
            feati, nesti, criti, depthi, featmaxi = np.unravel_index(np.argmax(meanPerformance), meanPerformance.shape)
            F = nFeatures[feati]
            E = estimators[nesti]
            C = criteria[criti]
            D = depths[depthi]
            M = featureMax[featmaxi]
            clf = RandomForestClassifier(n_estimators=E, criterion=C, max_depth=D, max_features=M, random_state=42)

        else:
            feati, alphai = np.unravel_index(np.argmax(meanPerformanceLR), meanPerformanceLR.shape)
            F = nFeatures[feati]
            A = alphas[alphai]
            clf = LogisticRegression(penalty='l2', C=A, random_state=42)


        if args.dataset == 'medseq':
            tt, pp, featureInd = featureSelection(np.array(Xtrain), ytrain, Ctrain[:,:2], absolute=1-args.pos)
            if args.pos:
                featureInd =  np.array([fi for fi in featureInd if tt[fi]>0])

        elif args.dataset == 'demographic':
            featureInd = np.arange(Xtrain.shape[1])
        else:
            assert args.dataset == 'both'
            featureInd = featureSelection(np.array(Xtrain), ytrain, np.zeros((Xtrain.shape[0],0)), absolute=1-args.pos)[2]

        cc = Xtrain.columns
        ss = StandardScaler()
        Xtrain = pd.DataFrame(ss.fit_transform(Xtrain), columns=cc)
        Xtest = pd.DataFrame(ss.transform(Xtest), columns=cc)

        XtrainF = Xtrain.iloc[:, featureInd[:F]]
        XtestF = Xtest.iloc[:, featureInd[:F]]


        clf.fit(XtrainF, ytrain)

        # print('\nTest')
        pred = clf.predict(XtestF)
        print('PPV %.3f' % precision_score(ytest, pred))
        print('ACC %.3f' % accuracy_score(ytest, pred))

        post = clf.predict_proba(XtestF)[:,1]
        [fpr, tpr, foldThresholdsROC] = roc_curve(ytest, post)
        ax[0].plot(fpr, tpr, color='C0', alpha=0.7, linestyle=lls[i], label=('Fold %d' % i))

        [pr, rc, foldThresholdsPR] = precision_recall_curve(ytest, post)
        ax[1].plot(rc, pr, color='C0', alpha=0.7, linestyle=lls[i], label=('Fold %d' % i))

        (npv, ppv), (spc, sns), _, _ = precision_recall_fscore_support(ytest,pred, average=None)
        acc = accuracy_score(ytest, pred)
        outerPerformance[i] = [npv, ppv, spc, sns, acc, roc_auc_score(ytest, post), average_precision_score(ytest, post)]


        #predictionArchive.append(predictions)
        posteriorArchive.append(post)

        #print(precision_recall_fscore_support(ytest, predictions, average='binary'))
        # print(precision_recall_fscore_support(ytest, predictions, average=None))

        allthresholds = np.union1d(allthresholds, foldThresholdsROC)
        allthresholds = np.union1d(allthresholds, foldThresholdsPR)

        for j in range(nperm):
            yrnd = np.random.permutation(ytest)

            (npv, ppv), (spc, sns), _, _ = precision_recall_fscore_support(yrnd,pred, average=None)
            acc = accuracy_score(yrnd, pred)
            outerPerformanceRnd[i,j] = [npv, ppv, spc, sns, acc, roc_auc_score(yrnd, post), average_precision_score(yrnd, post)]



    mu = np.mean(outerPerformance, 0)
    stderr = np.std(outerPerformance, axis=0, ddof=1) / np.sqrt(3)

    muRnd = np.mean(outerPerformanceRnd, axis=0)


    pvals = np.zeros(mu.shape)
    for i in range(mu.shape[0]):
        pvals[i] = np.mean(muRnd[:,i] >= mu[i], axis=0)

    nn = ['NPV', 'PPV', 'Specificity', 'Sensitivity', 'Accuracy', 'ROCAUC', 'AUPRC']
    xx = np.linspace(0,1,5)

    for j in range(len(nn)):
        print('%15s:\t%.3f +/- %.3f, p-value = %.4f' % (nn[j], mu[j], stderr[j], pvals[j]))



    ax[0].plot(xx,xx,'k--')

    ax[0].set_xlabel('False positive rate')
    ax[0].set_ylabel('True positive rate')

    ax[1].axhline(np.mean(labels), color='k', linestyle='--')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlim(0,1.05)
    plt.tight_layout()

    if not os.path.exists('../results'):
        os.mkdir('../results/')
    with open('../results/posteriors/%s.pkl' % args.dataset, 'wb') as f:
        pickle.dump({'thr': allthresholds, 'y': labelArchive, 'post': posteriorArchive}, f)



    ###########################################################################################################################
    # select best model in a single CV
    performanceRF = np.zeros((3, len(nFeatures), len(estimators), len(criteria), len(depths), len(featureMax)))
    performanceLR = np.zeros((3, len(nFeatures), len(alphas)))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for i, (trainInd, testInd) in enumerate(cv.split(dataTPM, labels)):
        Xtrain = dataTPM.iloc[trainInd]
        ytrain = labels[trainInd]

        Xtest = dataTPM.iloc[testInd]
        ytest = labels[testInd]


        Ctrain = covs[trainInd]
        Ctest = covs[testInd]


        if args.dataset == 'medseq':
            tt, pp, featureInd = featureSelection(np.array(Xtrain), ytrain, Ctrain[:,:2], absolute=1-args.pos)
            if args.pos:
                featureInd =  np.array([fi for fi in featureInd if tt[fi]>0])
        elif args.dataset == 'demographic':
            featureInd = np.arange(Xtrain.shape[1])

        else:
            assert args.dataset == 'both'
            featureInd = featureSelection(np.array(Xtrain), ytrain, np.zeros((Xtrain.shape[0],0)), absolute=1-args.pos)[2]

        cc = Xtrain.columns
        ss = StandardScaler()
        Xtrain = pd.DataFrame(ss.fit_transform(Xtrain), columns=cc)
        Xtest = pd.DataFrame(ss.transform(Xtest), columns=cc)

        for k, nfeat in enumerate(nFeatures):
            XtrainF = Xtrain.iloc[:, featureInd[:nfeat]]
            XtestF = Xtest.iloc[:, featureInd[:nfeat]]

            for l, aa in enumerate(alphas):
                clf = LogisticRegression(penalty='l2', C=aa, random_state=42)
                clf.fit(XtrainF, ytrain)


                pred = clf.predict_proba(XtestF)[:,1]
                if args.maximizeAUPRC:
                    performanceLR[i,k,l] = average_precision_score(ytest, pred)
                else:
                    performanceLR[i,k,l] = roc_auc_score(ytest, pred)

            for l, nest in enumerate(estimators):
                for m, criterion in enumerate(criteria):
                    for n, dp in enumerate(depths):
                        for o, fm in enumerate(featureMax):
                            clf = RandomForestClassifier(n_estimators=nest, criterion=criterion, max_depth=dp, max_features=fm, random_state=42)
                            clf.fit(XtrainF, ytrain)

                            pred = clf.predict_proba(XtestF)[:,1]
                            if args.maximizeAUPRC:
                                performanceRF[i,k,l,m,n,o] = average_precision_score(ytest, pred)
                            else:
                                performanceRF[i,k,l,m,n,o] = roc_auc_score(ytest, pred)



    meanPerformance = np.mean(performanceRF,axis=0)
    meanPerformanceLR = np.mean(performanceLR,axis=0)

    if np.max(meanPerformance) > np.max(meanPerformanceLR):
        feati, nesti, criti, depthi, featmaxi = np.unravel_index(np.argmax(meanPerformance), meanPerformance.shape)
        F = nFeatures[feati]
        E = estimators[nesti]
        C = criteria[criti]
        D = depths[depthi]
        M = featureMax[featmaxi]
        clf = RandomForestClassifier(n_estimators=E, criterion=C, max_depth=D, max_features=M, random_state=42)

    else:
        feati, alphai = np.unravel_index(np.argmax(meanPerformanceLR), meanPerformanceLR.shape)
        F = nFeatures[feati]
        A = alphas[alphai]
        clf = LogisticRegression(penalty='l2', C=A, random_state=42)


    tt, pp, featureInd = featureSelection(np.array(dataTPM), labels, covs[:,:2], absolute=1-args.pos)
    if args.pos:
        featureInd =  np.array([fi for fi in featureInd if tt[fi]>0])


    ss = StandardScaler()
    XtrainF = pd.DataFrame(ss.fit_transform(dataTPM), columns=cc)
    XtrainF = XtrainF.iloc[:,featureInd[:F]]

    clf.fit(XtrainF, labels)

    selected = np.array(XtrainF.columns)

    tt = featureSelection(np.array(dataTPM[selected]), labels, covs[:,:2], absolute=1-args.pos)[0]


    if args.dataset == 'medseq':
        with open('../results/featureNames_classification.pkl', 'wb') as f:
            pickle.dump({'feats': selected, 'tvalues': tt}, f)


        ii = np.where(clf.feature_importances_ != 0)[0]
        selected = selected[ii]
        tt = tt[ii]

        orderedFeats = XtrainF.columns[np.argsort(clf.feature_importances_)][::-1][:selected.shape[0]]
        orderedImportances = np.sort(clf.feature_importances_)[::-1][:selected.shape[0]]

        with open('../data/cpgi_enhancers_promoters.pkl', 'rb') as f:
            mp = pickle.load(f)

        markers = pd.DataFrame(index=orderedFeats)
        markers['importance'] = orderedImportances
        for k,v in mp.items():
            markers[k] = markers.index.map(v)



        genebodyD = {'chr15:26670042-26670606': 'GABRB3', 'chr6:18122019-18122763': 'NHLRC1',
       'chr6:165334274-165334481': 'PDE10A', 'chr19:2643289-2643635': 'GNG7',
       'chr7:158425159-158428017': 'PTPRN2', 'chrY:11332328-11333652': '-',
       'chr22:19656513-19656717': '-', 'chr5:178302222-178302465': 'COL23A1',
       'chr9:109534158-109534467': '-', 'chr10:2315054-2315277': 'ENSG00000294151,LINC00701',
       'chr11:1291616-1291899': 'TOLLIP', 'chr16:90052407-90053113': '-',
       'chr20:63043723-63044465': 'LINC01056,LINC01749', 'chr12:4243756-4244063': 'ENSG00000298091',
       'chr1:2280581-2282442': 'SKI', 'chr4:722492-723188': 'PCGF3,PCGF3-AS2',
       'chr8:138877638-138878204': 'COL22A1', 'chr8:1437039-1437274': 'DLGAP2',
       'chr11:68386341-68386705': 'LRP5', 'chr7:157794019-157794420': 'PTPRN2',
       'chr1:235993573-235993971': 'NID1', 'chr15:28457434-28457704': 'ENSG00000307371',
       'chr21:9705051-9706116': '-', 'chr4:186371271-186372424': 'F11-AS1',
       'chr1:166988983-166989446': 'MAEL', 'chr17:40316706-40316949': 'RARA',
       'chr1:245673021-245673445': 'KIF26B', 'chr10:64170856-64171129': 'ENSG00000228566',
       'chr3:32819649-32819937': 'TRIM71', 'chr22:37024184-37024859': 'MPST',
       'chr16:285829-286034': 'PDIA2', 'chr16:48496624-48496898': '-',
       'chr3:184561406-184562933': 'EPHB3', 'chr2:3448575-3448811': 'TRAPPC12',
       'chr6:31660561-31660881': 'C6orf47-AS1,C6orf47', 'chr12:34166664-34167049': '-',
       'chr6:163834418-163834648': 'ENSG00000298077', 'chr22:46214227-46214892': 'PPARA',
       'chr2:8027892-8028156': 'LINC00299', 'chr16:304243-304446': 'AXIN1',
       'chr4:188106455-188106879': 'TRIML2', 'chr4:8586988-8587224': 'GPR78',
       'chr13:113804562-113805007': 'TMEM255B', 'chrX:49589410-49590023': 'GAGE2A',
       'chr19:2540908-2541140': 'GNG7', 'chr10:124615225-124615644': '-',
       'chr6:126643503-126643842': 'ENSG00000293110,ENSG00000302933,ENSG00000293084', 'chr11:518682-520297': 'ENSG00000289997',
       'chr5:116361437-116361892': '-'}

        markers['gene_body'] = markers.index.map(genebodyD)


    # selected = fs.get_feature_names_out()
    print('Selected %d features' % selected.shape[0])
    print(np.intersect1d(selected, dmrs2.index).shape[0])


    ####################################################################3
    # FU samples
    v3TPM = pd.DataFrame(ss.transform(datav3), columns=cc)

    v3TPM = v3TPM[XtrainF.columns]
    print(np.mean(clf.predict(v3TPM)))


    hbd = pd.read_csv('/home/stavros/Desktop/code/medseq-backup/miracle-meth/data/hbd_cpgi.csv', index_col=0)
    hbdtpm = counts2tpm(hbd.iloc[:27923].astype(int))
    hbdtpm = np.log(hbdtpm.T + 1)
    hbdtpm = pd.DataFrame(ss.transform(hbdtpm), columns=cc, index=hbdtpm.index)
    hbdtpm = hbdtpm[XtrainF.columns]
    print(np.mean(clf.predict(hbdtpm)))

    # ############################################################################################
    # # link to cell types
    # cc = fs.get_feature_names_out()
    # dataS = np.array(dataTPM[cc])
    # from scipy.stats import ttest_ind
    # tstats, _ = ttest_ind(dataS[labels==0], dataS[labels==1])
    #
    # tstats = pd.Series(tstats, index=cc)
    #
    # cellTypeAtlasDMR = cellTypeAtlas[tstats.index]
    # cellTypeAtlasDMR.dropna(axis=1, inplace=True)
    #
    # dmrs2noY = tstats.loc[cellTypeAtlasDMR.columns]
    #
    # rhos = np.zeros(cellTypeAtlas.shape[0])
    # ps = np.zeros(cellTypeAtlas.shape[0])
    # for i in range(rhos.shape[0]):
    #     rhos[i], ps[i] = spearmanr(dmrs2noY, cellTypeAtlasDMR.iloc[i])
    #
    #
    #
    # tissues = pd.Series(cellTypeAtlasDMR.index).apply(gettissue)
    # allTissues = sorted(tissues.unique())
    #
    # tissue2ind = dict()
    # for t in allTissues:
    #     tissue2ind[t] = np.where(tissues == t)[0]
    #
    # fig, ax = plt.subplots(1,1)
    # figMean, axMean = plt.subplots(1,1)
    #
    # ax.axvline(-0.5, color='k', linestyle='--')
    #
    # xtick = []
    # base = 0
    # for i,t in enumerate(allTissues):
    #     print(t)
    #     rr = rhos[tissue2ind[t]]
    #     n = rr.shape[0]
    #     ax.bar(np.arange(n)+base, rr, color=('C%d' % (i%10)), linewidth=0.05)
    #
    #     ax.axvline(base+n-0.5, color='k', linestyle='--')
    #     xtick.append((n//2) + base)
    #
    #     base += n
    #
    #     axMean.bar(i, np.mean(rr), yerr=np.std(rr, ddof=1), color=('C%d' % (i%10)))
    #     axMean.errorbar(i, np.mean(rr), yerr=np.std(rr, ddof=1), color='k')
    #
    # ax.set_xticks(xtick)
    # ax.set_xticklabels(allTissues, rotation=45)
    #
    # ax.set_ylabel('Correlation FC vs cell type methylation')
    #
    # axMean.set_xticks(np.arange(len(allTissues)))
    # axMean.set_xticklabels(allTissues, rotation=45)

    ##########################################################################
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    x2 = pca.fit_transform(XtrainF)

    fig, ax = plt.subplots(1,1)

    ax.scatter(x2[labels==0,0], x2[labels==0,1], color='C0', label='HC')

    pd2 = x2[labels==1]

    patids = dataTPM.iloc[labels==1].index
    lab = np.array(clinical.loc[patids]['v1_subtype_zCC'])

    ax.scatter(pd2[lab=='1_Mild-Motor',0], pd2[lab=='1_Mild-Motor',1], color='C1', label='PD-motor', marker='s')
    ax.scatter(pd2[lab=='2_Intermediate',0], pd2[lab=='2_Intermediate',1], color='C2', label='PD-inter', marker='s')
    ax.scatter(pd2[lab=='3_Diffuse-Malignant',0], pd2[lab=='3_Diffuse-Malignant',1], color='C3', label='PD-diff', marker='s')

    ax.legend()

    Xpd = XtrainF[labels==1]

    ind = np.where(np.logical_or((lab== '1_Mild-Motor'),(lab=='3_Diffuse-Malignant')))[0]

    Xpd = Xpd[ind]
    lab = lab[ind]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for i, (trainInd, testInd) in enumerate(cv.split(Xpd, lab)):
        Xtrain = Xpd[trainInd]
        ytrain = lab[trainInd]

        Xtest = Xpd[testInd]
        ytest = lab[testInd]

        clf = LinearSVC(max_iter=5000)
        clf.fit(Xtrain, ytrain)
        print('\n%d' % i)
        print(clf.score(Xtrain, ytrain))
        print(clf.score(Xtest, ytest))
