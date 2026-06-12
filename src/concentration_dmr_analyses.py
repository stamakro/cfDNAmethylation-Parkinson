import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, kstest, zscore, ttest_ind, ttest_rel, wilcoxon, kruskal, spearmanr, fisher_exact
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
import statsmodels.formula.api as smf
plt.style.use('seaborn-v0_8-paper')
import statsmodels.imputation.mice as mice
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import os.path
import pickle
import argparse

#################################################################
# Function for multiple imputation results
#################################################################
def print_mice_mixed(imp_mdf, param_names):
    print("Results after multiple imputation")
    alpha = 0.05
    use_t = False
    xname = None
    params = imp_mdf.params
    bse = imp_mdf.bse
    tvalues = imp_mdf.tvalues
    pvalues = imp_mdf.pvalues
    conf_int = imp_mdf.conf_int(alpha)

    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)

    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|',
                        '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                        '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    data.index = param_names
    print(data)

#################################################################
# Load data
#################################################################
with open('../data/demographics.pkl', 'rb') as f:
    demographics = pickle.load(f)

with open('../data/info.pkl', 'rb') as f:
    info = pickle.load(f)

# remove low quality samples and patients with uncertain PD diagnosis
info = info[info['QC_pass']==1]

# class labels, 1: PD, 0: CTRL
labels = np.array(pd.Series(info.index).apply(lambda x: 0 if 'CTRL' in x  else 1))

patientIDs = ['_'.join(c.split('_')[:2]) for c in info.index]

# covariates
covs = demographics.loc[patientIDs]

conc = np.zeros(covs.shape[0])
for i in range(covs.shape[0]):
    if labels[i] == 0:
        conc[i] = info.loc[covs.index[i]]['cfdnaConcentration']
    else:
        conc[i] = info.loc[covs.index[i]+'_BL']['cfdnaConcentration']
covs['concentration'] = conc
covs['concentration_log10'] = np.log10(covs['concentration'])
covs['pd'] = labels
covs['patient'] = patientIDs

with open('../data/methylation_cpgi.pkl', 'rb') as f:
    data = pickle.load(f)
dmrs = pd.read_csv('../results/selected_features_with_regels_genebody.csv', index_col=0)
covs['DMRsum'] = data.loc[:, dmrs.index].sum(axis=1).to_frame('DMRsum')

#################################################################
# Mixed models for plasma cfDNA concentration
#################################################################
columns = ['concentration_log10','age', 'bmi', 'sexMale', 'pd']
data = covs.loc[:, ['concentration_log10','age', 'bmi', 'sexMale', 'pd','patient']]
imp = mice.MICEData(data.loc[:, columns])
fml = "concentration_log10~1 + age + bmi + sexMale + pd"
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['patient']},fit_kwds={'method': ["powell", "lbfgs"]})
pd_imp_mdf = mice_mlm.fit(10, 100)
print_mice_mixed(pd_imp_mdf,['Intercept ','age', 'bmi', 'sexMale', 'pd','random intercept'])

#################################################################
# Mixed model of disease severity
#################################################################
data = covs.loc[covs['pd']==1, ['concentration_log10','age_at_onset', 'bmi', 'education_years', 'sexMale', 'time_since_diagnosis', 'LEDD','subtype','patient']]
columns = ['concentration_log10','age_at_onset', 'bmi', 'education_years', 'sexMale', 'time_since_diagnosis', 'LEDD','subtype']

fml = "concentration_log10~1 + age_at_onset + bmi + education_years + LEDD + sexMale + time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['patient']},fit_kwds={'method': ["powell", "lbfgs"]})
time_imp_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for time since diagnosis after multiple imputation")
print_mice_mixed(time_imp_mdf,['Intercept ','age', 'bmi', 'education_years','LEDD','sexMale', 'time_since_diagnosis', 'random intercept'])

fml = "concentration_log10~1 + age_at_onset + bmi + education_years + LEDD + sexMale + time_since_diagnosis + subtype + subtype:time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['patient']},fit_kwds={'method': ["powell", "lbfgs"]})
subtype_imp_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for disease subtype after multiple imputation")
print_mice_mixed(subtype_imp_mdf,['Intercept ','age', 'bmi', 'education_years','LEDD','sexMale', 'time_since_diagnosis', 'subtype', 'subtype:time_since_diagnosis','random intercept'])

#################################################################
# Mixed model of DMRs
#################################################################
data = covs.loc[covs['pd']==1, ['DMRsum','age_at_onset', 'BMI', 'education_years', 'sex', 'time_since_diagnosis', 'LEDD','subtype','pom_id']]

columns = ['DMRsum','age_at_onset', 'BMI', 'education_years', 'sex', 'time_since_diagnosis', 'LEDD','subtype']
fml = "DMRsum~1 + age_at_onset + BMI + education_years + LEDD + sex + time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['pom_id']},fit_kwds={'method': ["powell", "lbfgs"]})
dmr_time_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for time since diagnosis after multiple imputation")
print_mice_mixed(dmr_time_mdf,['Intercept ','age', 'BMI', 'education_years','LEDD','sex', 'time_since_diagnosis', 'random intercept'])

fml = "DMRsum~1 + age_at_onset + BMI + education_years + LEDD + sex + time_since_diagnosis + subtype + subtype:time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['pom_id']},fit_kwds={'method': ["powell", "lbfgs"]})
subtype_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for disease subtype after multiple imputation")
print_mice_mixed(subtype_mdf,['Intercept ','age', 'BMI', 'education_years','LEDD','sex', 'time_since_diagnosis', 'subtype', 'subtype:time_since_diagnosis','random intercept'])
