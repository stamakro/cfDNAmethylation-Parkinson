import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import argparse
from statsmodels.stats.multitest import multipletests
import pickle
import sys
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--clinical', dest='clinical_data', default='')
parser.add_argument('--cfdna', dest='data', default='../data/methylation_cpgi.pkl')
parser.add_argument('--sample-info', dest='info', default='../data/info.pkl')
parser.add_argument('--target', dest='target', default='time')

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
info = info[~info['timepoint'].isna()]


dataTPM = data.loc[info.index]


pats1 = sorted(list(info[info['timepoint'] == 'BL'].index))
pats3 = sorted(list(info[info['timepoint'] == 'FU'].index))

allPatients = sorted(list(info['patient']))


clinical = clinical.loc[allPatients]


# standardize covariates
clinical['age_d_std'] = clinical['v1_Part1_Age'].astype(float) - (clinical['v1_time_since_diagnosis']/12.)
clinical['age_d_std'] -= clinical['age_d_std'].mean()
clinical['age_d_std'] /= clinical['age_d_std'].std()

clinical['bmi1_std'] = clinical['v1_Basaal_onderzoek_BodMasInd'].astype(float)
m = clinical['bmi1_std'].mean()
s = clinical['bmi1_std'].std()
clinical['bmi1_std'] -= m
clinical['bmi1_std'] /= s

clinical['bmi3_std'] = clinical['v3_Basaal_onderzoek_BodMasInd'].astype(float)
clinical['bmi3_std'] -= m
clinical['bmi3_std'] /= s


clinical['educ_std'] = clinical['v1_MOCA_NpsEducYears'].astype(float)
clinical['educ_std'] -= clinical['educ_std'].mean()
clinical['educ_std'] /= clinical['educ_std'].std()

clinical['v1_LEDD_std'] = clinical['v1_LEDD']
m = clinical['v1_LEDD_std'].mean()
s = clinical['v1_LEDD_std'].std()
clinical['v1_LEDD_std'] -= m
clinical['v1_LEDD_std'] /= s

clinical['v3_LEDD_std'] = clinical['v3_LEDD'] - m
clinical['v3_LEDD_std'] /= s

# make a long data frame with both time points
ages = []
sexes = []
bmis = []
ts = []
patids = []
edus = []
visits = []
samples = []
updrses = []
ledds = []

# make sex variable directly interpretable
nrToMF = {1.0: 'M', '0.0': 'F'}

for i in range(clinical.shape[0]):
    patid = clinical.index[i]
    age = clinical.iloc[i]['age_d_std']
    sex = nrToMF[clinical.iloc[i]['v1_Part1_Gender']]
    bmi1 = clinical.iloc[i]['bmi1_std']
    bmi3 = clinical.iloc[i]['bmi3_std']
    ey = clinical.iloc[i]['educ_std']
    sample1 = clinical.iloc[i]['v1_medseq_run']
    sample3 = clinical.iloc[i]['v3_medseq_run']

    # BL
    if patid in pats1.index:
        ages.append(age)
        sexes.append(sex)
        ts.append(clinical.iloc[i]['v1_time_since_diagnosis'])
        updrses.append(clinical.iloc[i]['v1_updrs3_OFF'])
        patids.append(patid)
        bmis.append(bmi1)
        edus.append(ey)
        visits.append(1)
        samples.append(sample1)
        ledds.append(clinical.iloc[i]['v1_LEDD_std'])

    # FU
    if patid in pats3.index:
        ages.append(age)
        sexes.append(sex)
        ts.append(clinical.iloc[i]['v3_time_since_diagnosis'])
        updrses.append(clinical.iloc[i]['v3_updrs3_OFF'])
        patids.append(patid)
        bmis.append(bmi3)
        edus.append(ey)
        visits.append(3)
        samples.append(sample3)
        ledds.append(clinical.iloc[i]['v1_LEDD_std'])

df = pd.DataFrame({'visit': visits, 'age_d': ages, 'sex': sexes, 'time': ts, 'updrs': updrses, 'patient': patids, 'BMI': bmis, 'education': edus, 'sample': samples, 'ledd': ledds})
print(df.shape)


# complete case analysis
df.dropna(axis=0, inplace=True)

# mixed effects model
coefsME = np.zeros(dataTPM.shape[1])
ppME = np.zeros(coefsME.shape[0])


for j, feat in tqdm(enumerate(dataTPM.columns), total=ppCS.shape[0]):
    profile = dataTPM[feat]

    profileLongForm = np.array(profile.loc[df['sample']])
    df['methyl'] = profileLongForm

    dfv1 = df[df['visit'] == 1]

    modelME = smf.mixedlm('%s ~ 1 + methyl + ledd + age_d + C(sex) + BMI + education' % args.target, data=df, groups=df['patient'],
                     re_formula='~1')  # random intercept
    resME = modelME.fit(method='nm')

    ppME[j] = resME.pvalues['methyl']
    coefsME[j] = resME.params['methyl']


results = pd.DataFrame({'beta_me': coefsME, 'p_me': ppME}, index=dataTPM.columns)

results['fdr_me'] = multipletests(ppME, method='fdr_bh')[1]

from datetime import datetime
today = str(datetime.now()).split(' ')[0]

results.to_csv('../results/%s_mixedmodel_disease_severity_%s.csv' % (today, args.target))


with open('../data/cpgi_enhancers_promoters.pkl', 'rb') as f:
    mp = pickle.load(f)

tt = results[results['fdr_me'] < 0.05]

for k, v in mp.items():
    tt[k] = tt.index.map(v)

gbd = {'chr14:64772604-64772948': 'SPTB', 'chr15:26670042-26670606': 'GABRB3', 'chr16:31148255-31148688': 'PRS33', 'chr3:51390981-51392168': 'RBM15B', 'chr6:18122019-18122763': 'NHLRC1'}
tt['gene_body'] = tt.index.map(gbd)
