#library(CAM)
library(mgcv)

source("utils/Rscript/train_gam.R")
source("utils/Rscript/selGam.R")
source("utils/Rscript/pruning.R")


dataset <- read.csv(file='{PATH_DATA}', header=FALSE, sep=",")
dag <- read.csv(file='{PATH_DAG}', header=FALSE, sep=",")
set.seed(42)
pruned_dag <- pruning(dataset, dag, pruneMethod = selGam, pruneMethodPars = list(cutOffPVal = {CUTOFF}, numBasisFcts = 10), output={VERBOSE})

write.csv(as.matrix(pruned_dag), row.names = FALSE, file = '{PATH_RESULTS}')
