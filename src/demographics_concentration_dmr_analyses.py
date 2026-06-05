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
from batch_icc import batch_icc
import pickle as pkl

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
# Read data files
#################################################################
ledd_pdq = pd.read_csv('../data/ledd_pdq.csv', index_col=0)
clin = pd.read_csv('../data/clinical_full_full.csv', index_col=0)
ctrl_clin = pd.read_csv('../data/cfDNA_controls_Demographics_19Feb2025.csv', index_col=0)

pd_medseq_sample = pd.read_csv('../data/sampleinfo260_cpg.csv', index_col=0)
pd_medseq_counts = pd.read_csv('../data/counts_aggregated_cpgi_pd.csv', index_col=0)
pd_cfdna_conc = pd.read_csv('../data/cfdna_concentration_pd.csv', index_col=0) 

ctrl_medseq_sample = pd.read_csv('../data/Samples-overview.csv', index_col=0)
ctrl_medseq_counts = pd.read_csv('../data/counts_aggregated_cpgi_controls.csv', index_col=0)
ctrl_cfdna_conc = pd.read_csv('../data/cfdna_concentration_control.csv', index_col=1)

pd_dmr_sums = pd.read_csv('../data/dmr_sums.csv', index_col=0)
ctrl_dmr_sums = pd.read_csv('../data/sumscores_ctrl.csv', index_col=0)

ctrl_disease_data = pd.read_csv('../data/cfDNA - Medische Aandoening_export_20251217.csv', delimiter=';', index_col=0)
file = '../data/diseases_v1andv3.pkl'
with open(file, mode='rb') as f:
    pd_diseases_data = pkl.load(f)
disease_classification = pd.read_csv('../data/disease_classification.csv', index_col=0)

#################################################################
# Preprocess data
#################################################################

clin = clin[clin['v1_Algemeen1_OFF_DiagParkCertain'] == '1.0']
clin = clin[clin['v3_diagnosisPD_DiagParkPersist'] == '1.0']
clin = clin.astype({'v1_Part1_Age': 'float','v1_time_since_diagnosis': 'float','v3_time_since_diagnosis': 'float'})

clin['v1_time_since_diagnosis'] = clin['v1_time_since_diagnosis']/12
clin['v2_time_since_diagnosis'] = clin['v2_time_since_diagnosis']/12
clin['v3_time_since_diagnosis'] = clin['v3_time_since_diagnosis']/12

time_between_measurements = (clin['v3_time_since_diagnosis']-clin['v1_time_since_diagnosis'])
time_between_measurements[time_between_measurements.isna()] = np.nanmedian(time_between_measurements)
v3_Part1_Age = clin['v1_Part1_Age'] + time_between_measurements

clin = pd.concat([clin, v3_Part1_Age.rename('v3_Part1_Age')], axis=1)
clin['v1_age_at_onset_PD_DiagParkPersist'] = clin['v1_Part1_Age'] - clin['v1_time_since_diagnosis']

pd_cfdna_clin = clin[~clin['v1_medseq_run'].isna()]
pd_rest_clin = clin[clin['v1_medseq_run'].isna()]

pd_cfdna_ledd_pdq = ledd_pdq[ledd_pdq['medseq']==1]
pd_rest_ledd_pdq = ledd_pdq[ledd_pdq['medseq']==0]

pd_cfdna_clin = pd_cfdna_clin.merge(pd_cfdna_ledd_pdq, left_index=True, right_index=True, how='left')
pd_rest_clin = pd_rest_clin.merge(pd_rest_ledd_pdq, left_index=True, right_index=True, how='left')

ctrl_cfdna_conc.set_index(ctrl_cfdna_conc.index.astype(str), inplace=True)
ctrl_medseq_sample.set_index(ctrl_medseq_sample.index.astype(str), inplace=True)
ctrl_clin.set_index(ctrl_clin.index.astype(str), inplace=True)

colContinuous = ['LEDDv1','PDQv1','v1_Part1_Age', 'v1_Basaal_onderzoek_BodMasInd', 'v1_time_since_diagnosis', 'v1_updrs3_OFF','v1_MOCA_NpsEducYears','v1_age_at_onset_PD_DiagParkPersist']
colCategorical = ['v1_Part1_Gender', 'v1_Hoehn__Yahr_stage_OFF_Up3OfHoeYah','v1_subtype_zMoCA']
optionsCategorical = {'v1_Part1_Gender': ['1.0', '2.0'], 'v1_Hoehn__Yahr_stage_OFF_Up3OfHoeYah': ['1.0', '2.0', '3.0', '4.0'],'v1_subtype_zMoCA': ['1_Mild-Motor', '2_Intermediate', '3_Diffuse-Malignant']}

#################################################################
# Prepare and cluster data
#################################################################
biomics2pom = {}
for k, v in pd_cfdna_clin['v1_medseq_run'].to_dict().items():
    biomics2pom[v] = k

for k, v in pd_cfdna_clin['v3_medseq_run'].to_dict().items():
    biomics2pom[v] = k

ctrl_clin['BMI'] = 1e4 * ctrl_clin['gewicht'] / (ctrl_clin['lengte']**2)
ctrl_clin = ctrl_clin.loc[ctrl_medseq_sample.index]
ctrl_cfdna_conc = ctrl_cfdna_conc.loc[ctrl_medseq_sample.index]
ctrl_medseq_sample['Used reads'] = ctrl_medseq_sample['CpG reads'] * ctrl_medseq_sample['Mapping'] * 0.01
ctrl_clin_m = ctrl_clin[ctrl_medseq_sample['CpG/Total'] > 20.]
ctrl_medseq_sample = ctrl_medseq_sample[ctrl_medseq_sample['CpG/Total'] > 20.]
ctrl_clin_m = ctrl_clin_m[ctrl_medseq_sample['Used reads'] > 3e6]
ctrl_medseq_sample = ctrl_medseq_sample[ctrl_medseq_sample['Used reads'] > 3e6]
assert (ctrl_clin_m.index == ctrl_medseq_sample.index).all()

pd_sample_info1 = pd_medseq_sample.loc[pd_cfdna_clin['v1_medseq_run']]
pd_sample_info3 = pd_medseq_sample.loc[pd_cfdna_clin['v3_medseq_run']]
pd_cfdna_clin1 = pd_cfdna_clin.loc[pd_sample_info1.index.map(biomics2pom)]
pd_cfdna_clin3 = pd_cfdna_clin.loc[pd_sample_info3.index.map(biomics2pom)]
pd_cfdna_clin13 = pd_cfdna_clin.loc[pd_sample_info1.index.map(biomics2pom).union(pd_sample_info3.index.map(biomics2pom))]

pd_cfdna_conc.loc[pd_cfdna_conc['plasma_cfdna_concentration']<=0, 'plasma_cfdna_concentration'] = np.nan
pd_concentration1 = pd_cfdna_conc.loc[pd_sample_info1.index]
pd_sample_info1 = pd_sample_info1[pd_sample_info1['Used reads'] > 3e6]
pd_sample_info1 = pd_sample_info1[pd_sample_info1['CpG/Total'] > 20.0]
pd_sample_info1 = pd_sample_info1[pd_sample_info1['notes'].isna()]
pd_cfdna_clinm1 = pd_cfdna_clin.loc[pd_sample_info1.index.map(biomics2pom)]

pd_concentration3 = pd_cfdna_conc.loc[pd_sample_info3.index]
pd_sample_info3 = pd_sample_info3[pd_sample_info3['Used reads'] > 3e6]
pd_sample_info3 = pd_sample_info3[pd_sample_info3['CpG/Total'] > 20.0]
pd_sample_info3 = pd_sample_info3[pd_sample_info3['notes'].isna()]
pd_cfdna_clinm3 = pd_cfdna_clin.loc[pd_sample_info3.index.map(biomics2pom)]

ctrl_medseq_counts = ctrl_medseq_counts[ctrl_medseq_sample.index]
xx = []
yy = []
def isFromChrom(cpg: str, chr: str):
    chrom = cpg.split(':')[0]
    if chrom == chr:
        return True
    return False
total = np.array(ctrl_medseq_counts.sum())
iiX = np.where(pd.Series(ctrl_medseq_counts.index).apply(isFromChrom,  chr='chrX'))[0]
totalX = ctrl_medseq_counts.iloc[iiX].sum() / total
iiY = np.where(pd.Series(ctrl_medseq_counts.index).apply(isFromChrom,  chr='chrY'))[0]
totalY = ctrl_medseq_counts.iloc[iiY].sum() / total
xx += list(totalX)
yy += list(totalY)
fig, ax = plt.subplots(1,1)
ax.scatter(totalX[ctrl_clin['geslacht'] == 2], totalY[ctrl_clin['geslacht'] == 2], color=list(plt.rcParams['axes.prop_cycle'])[0]['color'], label='F')
ax.scatter(totalX[ctrl_clin['geslacht'] == 1], totalY[ctrl_clin['geslacht'] == 1], color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], label='M')
pdData1 = pd_medseq_counts[pd_sample_info1.index]
total = np.array(pdData1.sum())
totalX = pdData1.iloc[iiX].sum() / total
totalX = np.array(totalX)
totalY = pdData1.iloc[iiY].sum() / total
totalY = np.array(totalY)
xx += list(totalX)
yy += list(totalY)
ax.scatter(totalX[pd_cfdna_clinm1['v1_Part1_Gender'] == '2.0'], totalY[pd_cfdna_clinm1['v1_Part1_Gender'] == '2.0'], color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
ax.scatter(totalX[pd_cfdna_clinm1['v1_Part1_Gender'] == '1.0'], totalY[pd_cfdna_clinm1['v1_Part1_Gender'] == '1.0'], color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
pdData3 = pd_medseq_counts[pd_sample_info3.index]
total = np.array(pdData3.sum())
totalX = pdData3.iloc[iiX].sum() / total
totalX = np.array(totalX)
totalY = pdData3.iloc[iiY].sum() / total
totalY = np.array(totalY)
xx += list(totalX)
yy += list(totalY)
ax.scatter(totalX[pd_cfdna_clinm3['v1_Part1_Gender'] == '2.0'], totalY[pd_cfdna_clinm3['v1_Part1_Gender'] == '2.0'], color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
ax.scatter(totalX[pd_cfdna_clinm3['v1_Part1_Gender'] == '1.0'], totalY[pd_cfdna_clinm3['v1_Part1_Gender'] == '1.0'], color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
ax.legend()
ax.set_xlabel('chrX')
ax.set_ylabel('chrY')
ds = np.vstack((xx,yy)).T
mix = GaussianMixture(n_components=2).fit(ds)
npoints = 200
x = np.linspace(0.9*np.min(xx), 1.1*np.max(xx), npoints)
y = np.linspace(0.9*np.min(yy), 1.1*np.max(yy), npoints)
for i, (m,C) in enumerate(zip(mix.means_, mix.covariances_)):
    f = np.zeros((npoints,npoints))
    for j in range(npoints):
        for k in range(npoints):
            f[j,k] = multivariate_normal.pdf([x[j], y[k]], m, C)

    ax.contour(x,y,f.T,colors=list(plt.rcParams['axes.prop_cycle'])[2]['color'], alpha=0.7)
ax.grid()
ax.set_ylim(0.004, 0.018)

fig.savefig("../results/figure_s1.svg", bbox_inches='tight')

#################################################################
# Other diseases
#################################################################
disease_classification = disease_classification[(disease_classification['disease_of_interest']!='Other') & (disease_classification['disease_of_interest']!='Sleep-wake disorders')]
dictionary = disease_classification.to_dict(orient='dict')['disease_of_interest']
unique_diseases = set(dictionary.values())

pd_diseases_v1 = pd.DataFrame(0, index=pd_diseases_data['visit1'].keys(), columns=list(unique_diseases))
ctrl_diseases = pd.DataFrame(0, index=ctrl_clin.index, columns=list(unique_diseases))

for x in pd_diseases_data['visit1']:
  for y in pd_diseases_data['visit1'][x]:
    if y in dictionary.keys():
      pd_diseases_v1.at[str(x), dictionary[y]] = 1

pd_diseases_v3 = pd_diseases_v1.copy()
for x in pd_diseases_data['visit3']:
  for y in pd_diseases_data['visit3'][x]:
    if y in dictionary.keys():
      pd_diseases_v3.at[str(x), dictionary[y]] = 1

for index, row in ctrl_disease_data.iterrows():
  if row['naamAandoening'] in dictionary.keys():
    ctrl_diseases.at[str(index), dictionary[row['naamAandoening']]] = 1
    
#################################################################
# Drop samples and add keys
#################################################################
pd_cfdna_clin1.drop(index=['POMU38588D7F10CCC56F'], inplace=True)
pd_cfdna_clin3.drop(index=['POMU38588D7F10CCC56F'], inplace=True)
pd_cfdna_clin13.drop(index=['POMU38588D7F10CCC56F'], inplace=True)
pd_sample_info1.drop(index=['I23-1219-09'], inplace=True)
pd_sample_info3.drop(index=['I23-1223-07'], inplace=True)
pd_concentration1.drop(index=['I23-1219-09'], inplace=True)
pd_concentration3.drop(index=['I23-1223-07'], inplace=True)
pd_concentration1['pom_id'] = pd_concentration1.index.map(biomics2pom)
pd_concentration3['pom_id'] = pd_concentration3.index.map(biomics2pom)
pd_cfdna_clin1['sample_id'] = pd_concentration1.index
pd_cfdna_clin3['sample_id'] = pd_concentration3.index
pd_cfdna_clin1['pom_id'] = pd_cfdna_clin1.index
pd_cfdna_clin3['pom_id'] = pd_cfdna_clin3.index

diff = set(pd_diseases_v1.index).difference(set(pd_cfdna_clin1.index))
pd_diseases_v1 = pd_diseases_v1.drop(index=diff)
pd_diseases_v3 = pd_diseases_v3.drop(index=diff)

#################################################################
# Table 1. patient cohorts
#################################################################

for c in colContinuous:
    print(c)
    vals = []
    for df, name in zip([pd_cfdna_clin13, pd_rest_clin], ['medseq', 'remaining']):

        print(name)
        val = np.array(df[c])
        val[val == '##USER_MISSING_95##'] = np.nan
        val = val.astype(float)

        print(np.nanpercentile(val, [25,50,75]))
        print('missing: %d, %.1f percent' % (np.sum(np.isnan(val)), 100*np.mean(np.isnan(val))))
        vals.append(val)

    if kstest(zscore(vals[0]),'norm').pvalue<0.05 or kstest(zscore(vals[1]),'norm').pvalue<0.05:
        print('\np-value=%.6f' % mannwhitneyu(vals[0], vals[1], nan_policy='omit')[1])
    else:
        print('\np-value=%.6f' % ttest_ind(vals[0], vals[1], nan_policy='omit')[1])        
    print('\n\n')

# gender: 1 = M, 2 = F
for c in colCategorical:
    print(c)
    mat = np.zeros((2,len(optionsCategorical[c])))
    missing = np.zeros((len(optionsCategorical[c])))
    for i, (df, name) in enumerate(zip([pd_cfdna_clin13, pd_rest_clin], ['medseq', 'remaining'])):

        # print(name)
        uniq = optionsCategorical[c]
        for j,k in enumerate(uniq):
            mat[i,j] = (df[c] == k).sum()
        missing[i] = (len(df[c]) - np.sum(mat[i,:]))/len(df[c])*100


    mat = pd.DataFrame(mat, index=['medseq', 'remaining'], columns=optionsCategorical[c])
    print(mat)
    if len(mat.columns) > 2:
        print('p-value = %.10f' % (chi2_contingency(mat)[1]))
    else:
        print('p-value = %.10f' % (fisher_exact(mat)[1]))

    print('Missing: MeD-seq = %.1f percent Remaining = %.1f percent' % (missing[0], missing[1]))
    print('\n\n')

#################################################################
# Table 2. case vs controls
#################################################################
PPP_equivalent_1 = {'leeftijd': 'v1_Part1_Age', 'BMI': 'v1_Basaal_onderzoek_BodMasInd'}
PPP_equivalent_3 = {'leeftijd': 'v3_Part1_Age', 'BMI': 'v3_Basaal_onderzoek_BodMasInd'}
for c in ['leeftijd', 'BMI']:
    print(c)
    vals = []
    print('controls')
    val = np.array(ctrl_clin[c])
    val = val.astype(float)
    print(np.nanpercentile(val, [25,50,75]))
    vals.append(val)

    print('PD - baseline')
    val = np.array(pd_cfdna_clin1[PPP_equivalent_1[c]])
    val[val == '##USER_MISSING_95##'] = np.nan
    val = val.astype(float)
    print(np.nanpercentile(val, [25,50,75]))
    vals.append(val)

    if kstest(zscore(vals[0]),'norm').pvalue<0.05 or kstest(zscore(vals[1]),'norm').pvalue<0.05:
        print('p-value=%.6f' % mannwhitneyu(vals[0], vals[1], nan_policy='omit')[1])
    else:
        print('p-value=%.6f' % ttest_ind(vals[0], vals[1], nan_policy='omit')[1])
    print('\n')
    
    print('PD - follow-up')
    val = np.array(pd_cfdna_clin3[PPP_equivalent_3[c]])
    val[val == '##USER_MISSING_95##'] = np.nan
    val = val.astype(float)
    print(np.nanpercentile(val, [25,50,75]))
    vals.append(val)
    
    if kstest(zscore(vals[0]),'norm').pvalue<0.05 or kstest(zscore(vals[2]),'norm').pvalue<0.05:
        print('p-value=%.6f' % mannwhitneyu(vals[0], vals[2], nan_policy='omit')[1])
    else:
        print('p-value=%.6f' % ttest_ind(vals[0], vals[2], nan_policy='omit')[1])
    print('\n')

print('sex')
matrix = [[np.sum(pd_cfdna_clin1['v1_Part1_Gender']=='1.0'), np.sum(pd_cfdna_clin1['v1_Part1_Gender']=='2.0')], [np.sum(ctrl_clin['geslacht']==1), np.sum(ctrl_clin['geslacht']==2)]]
print(matrix)
print(fisher_exact(matrix)[1])
matrix = [[np.sum(pd_cfdna_clin3['v1_Part1_Gender']=='1.0'), np.sum(pd_cfdna_clin3['v1_Part1_Gender']=='2.0')], [np.sum(ctrl_clin['geslacht']==1), np.sum(ctrl_clin['geslacht']==2)]]
print('\n')

print('Successful MeD-seq analysis - baseline')
matrix = [[len(pd_sample_info1), len(pd_cfdna_clin1)-len(pd_sample_info1)], [len(ctrl_medseq_sample), len(ctrl_clin)-len(ctrl_medseq_sample)]]
print(matrix)
print(fisher_exact(matrix)[1])
print('\n')

print('Successful MeD-seq analysis - follow-up')
matrix = [[len(pd_sample_info3), len(pd_cfdna_clin3)-len(pd_sample_info3)], [len(ctrl_medseq_sample), len(ctrl_clin)-len(ctrl_medseq_sample)]]
print(matrix)
print(fisher_exact(matrix)[1])
print('\n')

print('Mapping')
print('controls')
print(np.nanpercentile(ctrl_medseq_sample['Mapping'], [25,50,75]))
print('missing: %d, %.1f percent' % (np.sum(np.isnan(ctrl_medseq_sample['Mapping'])), 100*np.mean(np.isnan(ctrl_medseq_sample['Mapping']))))

print('PD - baseline')
print(np.nanpercentile(pd_sample_info1['Mapping'], [25,50,75]))
if kstest(zscore(pd_sample_info1['Mapping']),'norm').pvalue<0.05 or kstest(zscore(ctrl_medseq_sample['Mapping']),'norm').pvalue<0.05:
    print('p-value=%.6f' % mannwhitneyu(pd_sample_info1['Mapping'], ctrl_medseq_sample['Mapping'], nan_policy='omit')[1])
else:
    print('p-value=%.6f' % ttest_ind(pd_sample_info1['Mapping'], ctrl_medseq_sample['Mapping'], nan_policy='omit')[1])
print('\n')

print('PD - follow-up')
print(np.nanpercentile(pd_sample_info3['Mapping'], [25,50,75]))
if kstest(zscore(pd_sample_info3['Mapping']),'norm').pvalue<0.05 or kstest(zscore(ctrl_medseq_sample['Mapping']),'norm').pvalue<0.05:
    print('p-value=%.6f' % mannwhitneyu(pd_sample_info3['Mapping'], ctrl_medseq_sample['Mapping'], nan_policy='omit')[1])
else:
    print('p-value=%.6f' % ttest_ind(pd_sample_info3['Mapping'], ctrl_medseq_sample['Mapping'], nan_policy='omit')[1])
print('\n')

print('cfDNA concentrations - Success')
print('Baseline')
matrix = [[np.sum(np.isnan(pd_concentration1['plasma_cfdna_concentration'])), np.sum(~np.isnan(pd_concentration1['plasma_cfdna_concentration']))],
          [np.sum(np.isnan(ctrl_cfdna_conc['plasma_cfdna_concentration'])), np.sum(~np.isnan(ctrl_cfdna_conc['plasma_cfdna_concentration']))]]
print(matrix)
print(fisher_exact(matrix)[1])
print('\n')

print('Follow-up')
matrix = [[np.sum(np.isnan(pd_concentration3['plasma_cfdna_concentration'])), np.sum(~np.isnan(pd_concentration3['plasma_cfdna_concentration']))],
          [np.sum(np.isnan(ctrl_cfdna_conc['plasma_cfdna_concentration'])), np.sum(~np.isnan(ctrl_cfdna_conc['plasma_cfdna_concentration']))]]
print(matrix)
print(fisher_exact(matrix)[1])
print('\n')

print('Other diseases in PD-cfDNA and controls')
for disease in unique_diseases:
    matrix = [[np.sum(pd_diseases_v1[disease]), np.sum(pd_diseases_v1[disease]==0)], [np.sum(ctrl_diseases[disease]), np.sum(ctrl_diseases[disease]==0) ]]
    print(disease)
    print(matrix)
    print(fisher_exact(matrix)[1])
    print('\n')

for disease in unique_diseases:
    matrix = [[np.sum(pd_diseases_v3[disease]), np.sum(pd_diseases_v3[disease]==0)], [np.sum(ctrl_diseases[disease]), np.sum(ctrl_diseases[disease]==0) ]]
    print(disease)
    print(matrix)
    print(fisher_exact(matrix)[1])
    print('\n')

print('controls')
print(ctrl_diseases.sum(axis=1).value_counts())
print('PD - baseline')
print(pd_diseases_v1.sum(axis=1).value_counts())
print(chi2_contingency([[pd_diseases_v1.sum(axis=1).value_counts()], [ctrl_diseases.sum(axis=1).value_counts()]])[1])
print('\n')

print('PD - follow-up')
print(pd_diseases_v3.sum(axis=1).value_counts())
print(chi2_contingency([[pd_diseases_v3.sum(axis=1).value_counts()], [ctrl_diseases.sum(axis=1).value_counts()]])[1])
print('\n')

#################################################################
# Create dataframes for modeling
#################################################################
pd_data1 = pd_cfdna_clin1.loc[:,['sample_id','v1_Part1_Age','v1_Basaal_onderzoek_BodMasInd','v1_Part1_Gender','pom_id','v1_time_since_diagnosis',
                              'v1_MOCA_NpsEducYears','v1_age_at_onset_PD_DiagParkPersist','v1_subtype_zMoCA','LEDDv1']]
pd_data1 = pd.concat([pd_data1,pd_dmr_sums.loc[:,'DMRsum_v1']], axis=1)
pd_data1.rename(columns={"v1_Part1_Age": "age", "v1_Basaal_onderzoek_BodMasInd": "BMI", "v1_Part1_Gender": "sex", 'v1_subtype_zMoCA': 'subtype',
                      "v1_time_since_diagnosis": "time_since_diagnosis","v1_MOCA_NpsEducYears": "education_years",'v1_age_at_onset_PD_DiagParkPersist': 'age_at_onset',
                      'LEDDv1': 'LEDD','DMRsum_v1': 'DMRsum'}, inplace=True)
pd_data3 = pd_cfdna_clin3.loc[:,['sample_id','v3_Part1_Age','v3_Basaal_onderzoek_BodMasInd','v1_Part1_Gender','pom_id','v3_time_since_diagnosis',
                              'v3_MOCA_NpsEducYears','v1_age_at_onset_PD_DiagParkPersist','v1_subtype_zMoCA','LEDDv3']]
pd_data3 = pd.concat([pd_data3,pd_dmr_sums.loc[:,'DMRsum_v3']], axis=1)
pd_data3.rename(columns={"v3_Part1_Age": "age", "v3_Basaal_onderzoek_BodMasInd": "BMI", "v1_Part1_Gender": "sex", 'v1_subtype_zMoCA': 'subtype',
                      "v3_time_since_diagnosis": "time_since_diagnosis","v3_MOCA_NpsEducYears": "education_years",'v1_age_at_onset_PD_DiagParkPersist': 'age_at_onset',
                      'LEDDv3': 'LEDD','DMRsum_v3': 'DMRsum'}, inplace=True)

pd_data1.set_index(pd_data1.sample_id, inplace=True)
pd_data1 = pd_data1.assign(v=1)
pd_data3.set_index(pd_data3.sample_id, inplace=True)
pd_data3 = pd_data3.assign(v=3)
pd_data = pd.concat([pd_data1, pd_data3])
pd_data.drop('sample_id', axis=1, inplace=True)
pd_data['subtype'] = pd_data['subtype'].map({'1_Mild-Motor': 0, '2_Intermediate': 1, '3_Diffuse-Malignant': 2, '4_Undefined': np.nan})

pd_concentration_data1 = pd_concentration1.loc[:,['plasma_cfdna_concentration','pom_id']]
pd_concentration_data3 = pd_concentration3.loc[:,['plasma_cfdna_concentration','pom_id']]
pd_concentration_data1.index.names = ["sample_id"]
pd_concentration_data3.index.names = ["sample_id"]
pd_data = pd.concat((pd_data, pd.concat([pd_concentration_data1.loc[:,'plasma_cfdna_concentration'],pd_concentration_data3.loc[:,'plasma_cfdna_concentration']])), axis=1)

pd_data[pd_data['plasma_cfdna_concentration']<=0] = np.nan
pd_data['plasma_cfdna_concentration_log10'] = np.log10(pd_data['plasma_cfdna_concentration'])
pd_data = pd_data.astype({'subtype':'float','age': 'float','sex': 'float','BMI': 'float','pom_id': 'str','LEDD':'float','age_at_onset':'float',
                    'time_since_diagnosis': 'float','plasma_cfdna_concentration':'float','plasma_cfdna_concentration_log10':'float','education_years':'float'})
pd_data['sex'] = (pd_data['sex']==1).astype(float)
pd_data = pd_data.assign(pd=1)

x_ctrl = ctrl_clin.loc[:,['leeftijd','BMI','geslacht']]
x_ctrl.rename(columns={"leeftijd": "age", "BMI": "BMI", "geslacht": "sex"}, inplace=True)
x_ctrl = x_ctrl.assign(pd=0)
x_ctrl['pom_id'] = x_ctrl.index
x_ctrl.index.names = ["sample_id"]
x_ctrl['sex'] = (x_ctrl['sex']==1).astype(float)

y_ctrl = ctrl_cfdna_conc.loc[:,["sample_id",'plasma_cfdna_concentration']]
ctrl_data = pd.concat([y_ctrl, x_ctrl], axis=1)
ctrl_data = ctrl_data.astype({'plasma_cfdna_concentration':'float','age': 'float','sex': 'float','BMI': 'float','pom_id': 'str'})
ctrl_data['plasma_cfdna_concentration_log10'] = np.log10(ctrl_data['plasma_cfdna_concentration'])
ctrl_data.drop('sample_id', axis=1, inplace=True)

#################################################################
# Mixed models for plasma cfDNA concentration
#################################################################
columns = ['plasma_cfdna_concentration_log10','age', 'BMI', 'sex', 'pd']
data = pd.concat([ctrl_data.loc[:, ['plasma_cfdna_concentration_log10','age', 'BMI', 'sex', 'pd','pom_id']], pd_data.loc[:, ['plasma_cfdna_concentration_log10','age', 'BMI', 'sex', 'pd','pom_id']]], axis=0)
imp = mice.MICEData(data.loc[:, columns])
fml = "plasma_cfdna_concentration_log10~1 + age + BMI + sex + pd"
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['pom_id']},fit_kwds={'method': ["powell", "lbfgs"]})
pd_imp_mdf = mice_mlm.fit(10, 100)
print_mice_mixed(pd_imp_mdf,['Intercept ','age', 'BMI', 'sex', 'pd','random intercept'])

#################################################################
# Mixed model of disease severity
#################################################################
data = pd_data.loc[:, ['plasma_cfdna_concentration_log10','age_at_onset', 'BMI', 'education_years', 'sex', 'time_since_diagnosis', 'LEDD','subtype','pom_id']]
columns = ['plasma_cfdna_concentration_log10','age_at_onset', 'BMI', 'education_years', 'sex', 'time_since_diagnosis', 'LEDD','subtype']

fml = "plasma_cfdna_concentration_log10~1 + age_at_onset + BMI + education_years + LEDD + sex + time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['pom_id']},fit_kwds={'method': ["powell", "lbfgs"]})
time_imp_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for time since diagnosis after multiple imputation")
print_mice_mixed(time_imp_mdf,['Intercept ','age', 'BMI', 'education_years','LEDD','sex', 'time_since_diagnosis', 'random intercept'])

fml = "plasma_cfdna_concentration_log10~1 + age_at_onset + BMI + education_years + LEDD + sex + time_since_diagnosis + subtype + subtype:time_since_diagnosis"
imp = mice.MICEData(data.loc[:, columns])
mice_mlm = mice.MICE(fml, sm.MixedLM, imp, init_kwds={'groups': data['pom_id']},fit_kwds={'method': ["powell", "lbfgs"]})
subtype_imp_mdf = mice_mlm.fit(10, 100)
print("Mixed Linear Model results for disease subtype after multiple imputation")
print_mice_mixed(subtype_imp_mdf,['Intercept ','age', 'BMI', 'education_years','LEDD','sex', 'time_since_diagnosis', 'subtype', 'subtype:time_since_diagnosis','random intercept'])

#################################################################
# Figure 1. boxplot plasma cfDNA concentration by subtype
#################################################################

#################################################################
# Panel A
#################################################################
colorlist = list(plt.rcParams['axes.prop_cycle'])
markerlist = ['D', 'o', '^']

plasma_cfdna_concentration = [
    ctrl_cfdna_conc[~np.isnan(ctrl_cfdna_conc['plasma_cfdna_concentration'])]['plasma_cfdna_concentration'].values,
    pd_concentration1[~np.isnan(pd_concentration1['plasma_cfdna_concentration'])]['plasma_cfdna_concentration'].values,
    pd_concentration3[~np.isnan(pd_concentration3['plasma_cfdna_concentration'])]['plasma_cfdna_concentration'].values,
]

plasma_cfdna_concentration_scatter = [
    ctrl_cfdna_conc['plasma_cfdna_concentration'].values,
    pd_concentration1['plasma_cfdna_concentration'].values,
    pd_concentration3['plasma_cfdna_concentration'].values,
]

ticks = ['Controls', 'PwPD\nBaseline', 'PwPD\nFollow-up']

fig, ax = plt.subplots(2,2,figsize=(10,8))
ax[0,0].set_ylabel('Plasma cfDNA concentration [ng/mL]')
ax[0,0].set_yscale('log')

bpl = ax[0,0].boxplot(plasma_cfdna_concentration, 
                     positions=np.arange(len(plasma_cfdna_concentration)), widths=0.6,
                     patch_artist=True,  # fill with color
                medianprops={'color': 'black', 'label': '_median_'}, 
                showfliers=False)
for i in range(len(plasma_cfdna_concentration)):
    ax[0,0].scatter([i],np.pow(10, np.mean(np.log10(plasma_cfdna_concentration[i]))), marker=markerlist[i], color='white', edgecolor='black', s=50, linewidth=1,zorder=4)

#fill with colors
colors = [c['color'] for c in [colorlist[4], colorlist[3], colorlist[3]]]
for i, patch in enumerate(bpl['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.9)
    
x = plasma_cfdna_concentration_scatter.copy()
np.random.seed(198908)
for i in range(len(plasma_cfdna_concentration_scatter)):
    #y[i,:] = plasma_cfdna_concentration[i]
    x[i] = np.random.normal(i, 0.1, size=len(plasma_cfdna_concentration_scatter[i]))
    ax[0,0].scatter(x[i], plasma_cfdna_concentration_scatter[i], color='black', alpha=0.2, s=30,zorder=2, marker=markerlist[i], edgecolor='white', linewidth=1)

for i in range(len(plasma_cfdna_concentration_scatter[1])):
    ax[0,0].plot([x[1][i], x[2][i]], [plasma_cfdna_concentration_scatter[1][i], plasma_cfdna_concentration_scatter[2][i]], color='black', alpha=0.1, zorder=2)

ax[0,0].grid()

y, h, col = 1200, 200, 'k'
x1, x2 = 0, 1.5
ax[0,0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
ax[0,0].text((x1+x2)*.5, y+h, f"$\\it{{p}}={pd_imp_mdf.pvalues[4]:.3f}$", ha='center', va='bottom', color=col,fontsize=8)

y, h = 1100, 100
x1, x2 = 1, 2
ax[0,0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)

ax[0,0].set_xticks(np.arange(len(plasma_cfdna_concentration)), ticks)

#################################################################
# Panel B
#################################################################
concentration_data1 = pd_data.loc[pd_data['v']==1, ['time_since_diagnosis', 'plasma_cfdna_concentration']]
concentration_data3 = pd_data.loc[pd_data['v']==3, ['time_since_diagnosis', 'plasma_cfdna_concentration']]

ax[0,1].set_xlabel('Post-diagnostic disease duration [years]')
ax[0,1].set_yscale('log')
ax[0,1].set_xlim(0, 9)
ax[0,1].grid()

for i in range(len(concentration_data1)):
    ax[0,1].plot([concentration_data1.iloc[i,0], concentration_data3.iloc[i,0]], [concentration_data1.iloc[i,1], concentration_data3.iloc[i,1]], color='black', alpha=0.1,zorder=3)
    ax[0,1].scatter(concentration_data1.iloc[i,0], concentration_data1.iloc[i,1], color=colorlist[3]['color'], alpha=0.2,zorder=4,marker='o', edgecolor='white', s=30, linewidth=1)
    ax[0,1].scatter(concentration_data3.iloc[i,0], concentration_data3.iloc[i,1], color=colorlist[3]['color'], alpha=0.2,zorder=4,marker='^', edgecolor='white', s=30, linewidth=1)

sims = 1000
mean = time_imp_mdf.params[0:len(time_imp_mdf.params)-1]
cov_norm = time_imp_mdf.normalized_cov_params[0:len(time_imp_mdf.normalized_cov_params)-1,0:len(time_imp_mdf.normalized_cov_params)-1]
cov = cov_norm*np.outer(time_imp_mdf.bse[0:len(time_imp_mdf.bse)-1], time_imp_mdf.bse[0:len(time_imp_mdf.bse)-1])
x = np.random.multivariate_normal(mean, cov, sims)

grouped_data = pd_data.loc[:, ['age_at_onset','BMI','education_years','LEDD','sex']].mean().values

tt = np.linspace(0, 9, 100)
pred = np.zeros((sims, len(tt)))

for i in range(sims):
    for t in range(len(tt)):
        pred[i,t] = x[i][0] + np.sum(x[i][1:6]*grouped_data) + x[i][6]*tt[t]

ax[0,1].plot(tt, np.pow(10, np.mean(pred, axis=0)), color=colorlist[3]['color'],zorder=10,linewidth=2)
ax[0,1].fill_between(tt, np.pow(10, np.percentile(pred, 2.5, axis=0)), np.pow(10, np.percentile(pred, 97.5, axis=0)), color=colorlist[3]['color'], alpha=0.2)
ax[0,1].scatter(np.mean(pd_cfdna_clin1.loc[:,'v1_time_since_diagnosis'], axis=0), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration[1]))), marker='o', color=colorlist[3]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[0,1].scatter(np.mean(pd_cfdna_clin1.loc[:,'v3_time_since_diagnosis'], axis=0), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration[2]))), marker='^', color=colorlist[3]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[0,1].text(5, 1000, f"$\\it{{\\beta_{{duration}}}}={pow(10, time_imp_mdf.params[6]):.3f}$", ha='left', va='bottom', fontsize=8)

#################################################################
# Panel C
#################################################################
pd_concentration1_mm = pd_data.loc[(pd_data['subtype']==0)&(pd_data['v']==1), 'plasma_cfdna_concentration'].values
pd_concentration1_intm = pd_data.loc[(pd_data['subtype']==1)&(pd_data['v']==1), 'plasma_cfdna_concentration'].values
pd_concentration1_diff = pd_data.loc[(pd_data['subtype']==2)&(pd_data['v']==1), 'plasma_cfdna_concentration'].values

pd_concentration3_mm = pd_data.loc[(pd_data['subtype']==0)&(pd_data['v']==3), 'plasma_cfdna_concentration'].values
pd_concentration3_intm = pd_data.loc[(pd_data['subtype']==1)&(pd_data['v']==3), 'plasma_cfdna_concentration'].values
pd_concentration3_diff = pd_data.loc[(pd_data['subtype']==2)&(pd_data['v']==3), 'plasma_cfdna_concentration'].values

plasma_cfdna_concentration1 = [
    pd_concentration1_mm[~np.isnan(pd_concentration1_mm)],
    pd_concentration1_intm[~np.isnan(pd_concentration1_intm)],
    pd_concentration1_diff[~np.isnan(pd_concentration1_diff)],
]
plasma_cfdna_concentration3 = [
    pd_concentration3_mm[~np.isnan(pd_concentration3_mm)],
    pd_concentration3_intm[~np.isnan(pd_concentration3_intm)],
    pd_concentration3_diff[~np.isnan(pd_concentration3_diff)],
]

plasma_cfdna_concentration1_scatter = [
    pd_concentration1_mm,
    pd_concentration1_intm,
    pd_concentration1_diff,
]
plasma_cfdna_concentration3_scatter = [
    pd_concentration3_mm,
    pd_concentration3_intm,
    pd_concentration3_diff,
]

ticks = ['Mild-motor\nPredominant', 'Intermediate', 'Diffuse\nMalignant']

ax[1,0].set_ylabel('Plasma cfDNA concentration [ng/mL]')
ax[1,1].set_xlabel('Post-diagnostic disease duration [years]')
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')

bpl = ax[1,0].boxplot(plasma_cfdna_concentration1, 
                positions=np.arange(len(plasma_cfdna_concentration1))-0.25, widths=0.4,
                patch_artist=True,  # fill with color
                medianprops={'color': 'black', 'label': '_median_'}, 
                showfliers=False)

bp3 = ax[1,0].boxplot(plasma_cfdna_concentration3,
                positions=np.arange(len(plasma_cfdna_concentration3))+0.25, widths=0.4,
                patch_artist=True,  # fill with color
                medianprops={'color': 'black', 'label': '_median_'}, 
                showfliers=False)

for i in range(len(plasma_cfdna_concentration1_scatter)):
    y = plasma_cfdna_concentration1_scatter[i]
    x = np.random.normal(i, 0.05, size=len(y))
    ax[1,0].scatter(x-0.25, y, color='black', alpha=0.2, s=30,zorder=2, marker=markerlist[1], edgecolor='white', linewidth=1)
    y = plasma_cfdna_concentration3_scatter[i]
    ax[1,0].scatter(x+0.25, y, color='black', alpha=0.2, s=30,zorder=2, marker=markerlist[2], edgecolor='white', linewidth=1)
    ax[1,0].plot([x-0.25, x+0.25], [plasma_cfdna_concentration1_scatter[i], plasma_cfdna_concentration3_scatter[i]], color='black', alpha=0.1,zorder=3)
    ax[1,0].scatter(i-0.25,np.pow(10, np.mean(np.log10(plasma_cfdna_concentration1[i]))), marker=markerlist[1], color='white', edgecolor='black', s=50, linewidth=1,zorder=4)
    ax[1,0].scatter(i+0.25,np.pow(10, np.mean(np.log10(plasma_cfdna_concentration3[i]))), marker=markerlist[2], color='white', edgecolor='black', s=50, linewidth=1,zorder=4)

colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])[0:3]]
for i, patch in enumerate(bpl['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.9)

for i, patch in enumerate(bp3['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.9)

###
#Panel D
###
sims = 1000
mean = subtype_imp_mdf.params[0:len(subtype_imp_mdf.params)-1]
cov_norm = subtype_imp_mdf.normalized_cov_params[0:len(subtype_imp_mdf.normalized_cov_params)-1,0:len(subtype_imp_mdf.normalized_cov_params)-1]
cov = cov_norm*np.outer(subtype_imp_mdf.bse[0:len(subtype_imp_mdf.bse)-1], subtype_imp_mdf.bse[0:len(subtype_imp_mdf.bse)-1])
x = np.random.multivariate_normal(mean, cov, sims)

grouped_data = np.zeros((3, 5))
grouped_data[0,:] = data.loc[data['subtype']==0, ['age_at_onset','BMI','education_years','LEDD','sex']].mean().values
grouped_data[1,:] = data.loc[data['subtype']==1, ['age_at_onset','BMI','education_years','LEDD','sex']].mean().values
grouped_data[2,:] = data.loc[data['subtype']==2, ['age_at_onset','BMI','education_years','LEDD','sex']].mean().values

tt = np.linspace(0, 9, 100)
pred = np.zeros((3,sims, len(tt)))

for i in range(sims):
    for t in range(len(tt)):
        pred[0,i,t] = x[i][0] + np.sum(x[i][1:6]*grouped_data[0,:]) + x[i][6]*tt[t] + x[i][7]*0 + x[i][8]*0*tt[t]
        pred[1,i,t] = x[i][0] + np.sum(x[i][1:6]*grouped_data[1,:]) + x[i][6]*tt[t] + x[i][7]*1 + x[i][8]*1*tt[t]
        pred[2,i,t] = x[i][0] + np.sum(x[i][1:6]*grouped_data[2,:]) + x[i][6]*tt[t] + x[i][7]*2 + x[i][8]*2*tt[t]

ax[1,1].plot(tt, np.pow(10, np.mean(pred[0,:,:], axis=0)), color=list(plt.rcParams['axes.prop_cycle'])[0]['color'],zorder=10,linewidth=2)
ax[1,1].plot(tt, np.pow(10, np.mean(pred[1,:,:], axis=0)), color=list(plt.rcParams['axes.prop_cycle'])[1]['color'],zorder=10,linewidth=2)
ax[1,1].plot(tt, np.pow(10, np.mean(pred[2,:,:], axis=0)), color=list(plt.rcParams['axes.prop_cycle'])[2]['color'],zorder=10,linewidth=2)

ax[1,1].fill_between(tt, np.percentile(np.pow(10, pred[0,:,:]), 2.5, axis=0), np.percentile(np.pow(10, pred[0,:,:]), 97.5, axis=0), color=list(plt.rcParams['axes.prop_cycle'])[0]['color'], alpha=0.2)
ax[1,1].fill_between(tt, np.percentile(np.pow(10, pred[1,:,:]), 2.5, axis=0), np.percentile(np.pow(10, pred[1,:,:]), 97.5, axis=0), color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], alpha=0.2)
ax[1,1].fill_between(tt, np.percentile(np.pow(10, pred[2,:,:]), 2.5, axis=0), np.percentile(np.pow(10, pred[2,:,:]), 97.5, axis=0), color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], alpha=0.2)

for s in range(3):
    data_s = pd_data.loc[pd_data['subtype']==s, ['pom_id','time_since_diagnosis','plasma_cfdna_concentration']]
    subs = data_s['pom_id'].unique()
    for i in range(len(subs)):
        data_s[data_s['pom_id']==subs[0]]
        ax[1,1].plot(data_s.loc[data_s['pom_id']==subs[i],'time_since_diagnosis'], data_s.loc[data_s['pom_id']==subs[i], 'plasma_cfdna_concentration'], color='black', alpha=0.1,zorder=3)
        t = data_s.loc[data_s['pom_id']==subs[i],'time_since_diagnosis']
        c = data_s.loc[data_s['pom_id']==subs[i], 'plasma_cfdna_concentration']
        for j in range(len(t)):
            ax[1,1].scatter(t.iloc[j], c.iloc[j], color=list(plt.rcParams['axes.prop_cycle'])[s]['color'], alpha=0.2,zorder=4, marker=markerlist[j+1], edgecolor='white', linewidth=1, s=30)
        

ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==0)&(pd_data['v']==1), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10([plasma_cfdna_concentration1[0]]))), marker='o', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==0)&(pd_data['v']==3), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration3[0]))), marker='^', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==1)&(pd_data['v']==1), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration1[1]))), marker='o', color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==1)&(pd_data['v']==3), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration3[1]))), marker='^', color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==2)&(pd_data['v']==1), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration1[2]))), marker='o', color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1,1].scatter(pd_data.loc[(pd_data['subtype']==2)&(pd_data['v']==3), 'time_since_diagnosis'].mean(), np.pow(10, np.mean(np.log10(plasma_cfdna_concentration3[2]))), marker='^', color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)

ax[1,1].text(5, 1000, f"$\\it{{\\beta_{{subtype}}}}={pow(10, subtype_imp_mdf.params[7]):.3f}$", ha='left', va='bottom', color=col,fontsize=8)
ax[1,1].text(5, 600, f"$\\it{{\\beta_{{subtype*duration}}}}={pow(10, subtype_imp_mdf.params[8]):.3f}$", ha='left', va='bottom', color=col,fontsize=8)

ax[1,0].grid()
ax[1,0].set_xticks(np.arange(len(plasma_cfdna_concentration1)))
ax[1,0].set_xticklabels(ticks)
ax[1,0].sharey(ax[1,1])

ax[1,1].grid()
ax[1,1].set_xlim(0, 9)
ax[1,1].sharey(ax[1,0])

fig.savefig("../results/figure_1.svg", bbox_inches='tight')
fig.savefig("../results/figure_1.png", bbox_inches='tight')

#################################################################
# Mixed model of disease severity
#################################################################
data = pd_data.loc[:, ['DMRsum','age_at_onset', 'BMI', 'education_years', 'sex', 'time_since_diagnosis', 'LEDD','subtype','pom_id']]

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

#################################################################
# Figure 3. boxplot DMR sum score
#################################################################

#################################################################
# Panel A
#################################################################
colorlist = list(plt.rcParams['axes.prop_cycle'])
markerlist = ['D','v','o', '^']

pd_dmr_sum_scores = [
    ctrl_dmr_sums.loc[ctrl_dmr_sums['external']==0,'sumscore'].values,
    ctrl_dmr_sums.loc[ctrl_dmr_sums['external']==1,'sumscore'].values,
    pd_dmr_sums['DMRsum_v1'][~np.isnan(pd_dmr_sums['DMRsum_v1'])].values,
    pd_dmr_sums['DMRsum_v3'][~np.isnan(pd_dmr_sums['DMRsum_v3'])].values
]

pd_dmr_clin = pd_cfdna_clin1.loc[pd_cfdna_clin1.index.isin(pd_dmr_sums.index)]

pd_dmr_sum_scores_scatter = [
    ctrl_dmr_sums.loc[ctrl_dmr_sums['external']==0,'sumscore'].values,
    ctrl_dmr_sums.loc[ctrl_dmr_sums['external']==1,'sumscore'].values,
    pd_dmr_sums['DMRsum_v1'][~np.isnan(pd_dmr_sums['DMRsum_v1'])].values,
    pd_dmr_sums['DMRsum_v3'][~np.isnan(pd_dmr_sums['DMRsum_v3'])].values
]

ticks = ['Controls', 'External controls', 'PwPD\nBaseline', 'PwPD\nFollow-up']

fig, ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_ylabel('DMR sum score')

bpl = ax[0].boxplot(pd_dmr_sum_scores, 
                     positions=np.arange(len(pd_dmr_sum_scores)), widths=0.6,
                     patch_artist=True,  # fill with color
                medianprops={'color': 'black', 'label': '_median_'}, 
                showfliers=False)
for i in range(len(pd_dmr_sum_scores)):
    ax[0].scatter([i],np.mean(pd_dmr_sum_scores[i]), marker=markerlist[i], color='white', edgecolor='black', s=50, linewidth=1,zorder=4)

#fill with colors
colors = [c['color'] for c in [colorlist[4], colorlist[5],colorlist[3], colorlist[3]]]
for i, patch in enumerate(bpl['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.9)
    
x = pd_dmr_sum_scores_scatter.copy()
np.random.seed(198908)
for i in range(len(pd_dmr_sum_scores_scatter)):
    #y[i,:] = plasma_cfdna_concentration[i]
    x[i] = np.random.normal(i, 0.1, size=len(pd_dmr_sum_scores_scatter[i]))
    ax[0].scatter(x[i], pd_dmr_sum_scores_scatter[i], color='black', alpha=0.2, s=30,zorder=2, marker=markerlist[i], edgecolor='white', linewidth=1)

for i in range(len(pd_dmr_sum_scores_scatter[2])):
    ax[0].plot([x[2][i], x[3][i]], [pd_dmr_sum_scores_scatter[2][i], pd_dmr_sum_scores_scatter[3][i]], color='black', alpha=0.1, zorder=2)

ax[0].grid()
ax[0].set_xticks(np.arange(len(pd_dmr_sum_scores)), ticks)

#################################################################
# Panel B
#################################################################
time_since_diagnosis = pd_cfdna_clin1.loc[:,['v1_time_since_diagnosis','v3_time_since_diagnosis']]
sum_score_data = pd.concat([time_since_diagnosis,pd_dmr_sums], axis=1)

ax[1].set_xlabel('Post-diagnostic disease duration [years]')
ax[1].set_xlim(-0, 9)
ax[1].grid()

for i in range(len(sum_score_data)):
    ax[1].plot(sum_score_data.iloc[i,0:2], sum_score_data.iloc[i,2:4], color='black', alpha=0.1,zorder=3)
    ax[1].scatter(sum_score_data.iloc[i,0:1], sum_score_data.iloc[i,2:3], color=colorlist[3]['color'], alpha=0.2,zorder=4,marker='o', edgecolor='white', s=30, linewidth=1)
    ax[1].scatter(sum_score_data.iloc[i,1:2], sum_score_data.iloc[i,3:4], color=colorlist[3]['color'], alpha=0.2,zorder=4,marker='^', edgecolor='white', s=30, linewidth=1)

sims = 1000
mean = dmr_time_mdf.params[0:len(dmr_time_mdf.params)-1]
cov_norm = dmr_time_mdf.normalized_cov_params[0:len(dmr_time_mdf.normalized_cov_params)-1,0:len(dmr_time_mdf.normalized_cov_params)-1]
cov = cov_norm*np.outer(dmr_time_mdf.bse[0:len(dmr_time_mdf.bse)-1], dmr_time_mdf.bse[0:len(dmr_time_mdf.bse)-1])
x = np.random.multivariate_normal(mean, cov, sims)

grouped_data = data.loc[:, ['age_at_onset','BMI','education_years','LEDD','sex']].mean().values

tt = np.linspace(0, 9, 100)
pred = np.zeros((sims, len(tt)))

for i in range(sims):
    for t in range(len(tt)):
        pred[i,t] = x[i][0] + np.sum(x[i][1:6]*grouped_data) + x[i][6]*tt[t]

ax[1].plot(tt, np.mean(pred, axis=0), color=colorlist[3]['color'],zorder=10,linewidth=2)
ax[1].fill_between(tt, np.percentile(pred, 2.5, axis=0), np.percentile(pred, 97.5, axis=0), color=colorlist[3]['color'], alpha=0.2)
ax[1].scatter(np.mean(pd_cfdna_clin1.loc[:,'v1_time_since_diagnosis'], axis=0), np.mean(pd_dmr_sum_scores[2]), marker='o', color=colorlist[3]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1].scatter(np.mean(pd_cfdna_clin1.loc[:,'v3_time_since_diagnosis'], axis=0), np.mean(pd_dmr_sum_scores[3]), marker='^', color=colorlist[3]['color'], edgecolor='white',linewidth=1, s=50, zorder=11)
ax[1].text(5, 180, f"$\\it{{\\beta_{{duration}}}}={dmr_time_mdf.params[6]:.3f}$", ha='left', va='bottom', fontsize=8)
ax[1].sharey(ax[0])

fig.savefig("../results/figure_3.svg", bbox_inches='tight')
fig.savefig("../results/figure_3.png", bbox_inches='tight')