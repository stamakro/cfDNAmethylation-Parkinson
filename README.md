# cfDNAmethylation-Parkinson
Code to reproduce the manuscript "Proof-of-principle for cfDNA-based biomarkers of Parkinson’s disease progression" by Makrodimitris et al.

## Contributors
Stavros Makrodimitris,\
Rick van der Vliet

## Data
Processed cfDNA methylation data (TPM-normalized and log-transformed with a pseudo count of 1) are available in *data/* for CpG islands and cell type-specific hypermethylated regions.\
Sample information and cfDNA concentrations are also provided in *data/*. For the clinical data used in this project, contact the corresponding author.

# Scripts
This repository contains all necessary scripts to reproduce the work at *src/*.\
Access to clinical data is required for certain analyses.

# Dependencies
R 4.1\
Python 3.11\
numpy 1.26\
scipy 1.10\
sklearn 1.3\
pandas 2.1\
rpy2 3.5\
