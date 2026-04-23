import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
import pickle
import os
import sys

def getRegionLength(region: str) -> int:
    rr = region.split(':')[1].split('-')
    return int(rr[1]) - int(rr[0])

def counts2tpm(data: pd.DataFrame) -> pd.DataFrame:
    # dataframe features x samples
    # with index being chr:start-end
    # return a dataframe of the same size with tpm values

    regionSize = pd.Series(data.index).apply(getRegionLength)
    regionSize = pd.Series(np.array(regionSize), index=data.index)

    countsPerBase = data.T / regionSize
    total = countsPerBase.sum(axis=1)
    tpm = 1e6 * countsPerBase.T / total

    return tpm


def isAutosomal(region):
    autosomals = set([('chr%d' % i) for i in range(1,23)])

    return region.split(':')[0] in autosomals

def r_dmr_edgeR_standard(counts, librarysize, labels, removeLowCounts, covariates=None):
    """
    use edgeR to call DMRs, then perform FDR correction
    counts: a pd.DataFrame with shape (samples, genes)
    librarysize and labels should be np.ndarray's of shape (samples,)
    labels is assumed to have two different values (ie this finds DMRs between exactly 2 groups)
    removeLowCounts: bool whether to call filterByExpr
    covariates: things to include in the samples argument of DGEList

    # RETURNS: data frame with significant regions as rows """

    if covariates is None:
        covariates = 0


    with np_cv_rules.context():
        res = robjects.globalenv['dmrCallEdgeRstandard'](counts, librarysize, labels, removeLowCounts, covariates)


    return res



def r_dmr_edgeR_tissue(counts, librarysize, labels, maxMarkers=500, minFC=2.5):
    # use edgeR to call DMRs, then perform Bonferroni correction
    # at most maxMarkers genes are selected if significant at FWER=0.05 AND absolute foldchange > minFC
    # counts: a pd.DataFrame with shape (samples, genes)
    # librarysize and labels should be np.ndarray's of shape (samples,)
    # labels is assumed to have two different values (ie this finds DMRs between exactly 2 groups)

    # RETURNS: TODO!

    with np_cv_rules.context():
        res = robjects.globalenv['dmrCallEdgeR'](counts, librarysize, labels, maxMarkers, minFC)


    return res



def r_estimateMeanVariance(data, labels):
    with np_cv_rules.context():
        res = robjects.globalenv['estimateMeanVariance'](data, labels)


    return res


def r_estimateTF(data, mus, disps):
    with np_cv_rules.context():
        res = robjects.globalenv['estimateTF'](data, mus, disps)

    return res


def r_parseRemain(data, knownfracs, mus, disps, columnInd):

    fr = np.vstack((1-knownfracs, knownfracs))

    with np_cv_rules.context():
        res = robjects.globalenv['supervisedDeconvolution'](data, fr, mus, disps, columnInd)

    return res


def r_dispersions(data, df=10):
    with np_cv_rules.context():
        res = robjects.globalenv['getDispersions'](data, df)

    return res

np_cv_rules = default_converter + numpy2ri.converter + pandas2ri.converter

r_source = robjects.r['source']
#r_source('rutils.R')
r_source('/home/stavros/surf-snellius/Scripts/medseq/case-control/rutils.R')







if __name__ == '__main__':
    raise NotImplementedError
