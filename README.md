# DHT-CIT

## Introduction
This repository contains the implementation code for paper:

**Learning Causal Relations from Subsampled Time Series with Two Time-Slices** 

Anpeng Wu, Haoxuan Li, Kun Kuang, Keli Zhang, and Fei Wu

## Env:

### Hardware Configuration
Ubuntu 16.04.3 LTS operating system with 2 * Intel Xeon E5-2660 v3 @ 2.60GHz CPU (40 CPU cores, 10 cores per physical CPU, 2 threads per core), 256 GB of RAM, and 4 * GeForce GTX TITAN X GPU with 12GB of VRAM.

### Software Configuration
```shell
conda create -n causal python=3.8
conda activate causal
pip install cdt ylearn causal-learn seaborn pandas matplotlib GPy bokeh igraph scikit-learn networkx==2.8.5
conda install r-base=4.2.0
```

```shell
R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(version = "3.16")
BiocManager::install(c("GenomicFeatures", "AnnotationDbi"))
BiocManager::install(c("bnlearn"),force=TRUE)
BiocManager::install(c("mboost"),force=TRUE)
BiocManager::install(c("pcalg"),force=TRUE)
BiocManager::install(c("kpcalg"),force=TRUE)
BiocManager::install(c("RcppEigen","glmnet"),force=TRUE)
BiocManager::install(c("SID"),force=TRUE)
```

## Public Data:

[PM-CMR](https://pasteur.epa.gov/uploads/10.23719/1506014/SES_PM25_CMR_data.zip)
