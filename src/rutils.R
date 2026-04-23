library(edgeR)
library(MCMCpack) #for sampling from Dirichlet
library(pracma) #for convolution
library(Rsolnp) # for optimization with lagrange mult (eq constraint)

#source('./mark_estdistrpar.R')
#
#supervisedDeconvolution <- function(data, knownfracs, muTs, sizeTs, columnInd) {
#
#  datk = t(data)
#  Ngenes = dim(datk)[1]
#  print(Ngenes)
#  parsnt <- suppressWarnings(parsremain(datk, js=1:Ngenes, musingle=muTs[,columnInd,drop=FALSE],dispsingle = sizeTs[,columnInd,drop=FALSE],
#                                        initf=knownfracs, ngrid=25, optimizer="rsolnp",prior=FALSE,cleverinit=TRUE,distr ="nb"))
#  # Retrieve parameters
#  logmeans <- unlist(lapply(parsnt,function(el) if(class(el) =="try-error") return(NA) else return(el$pars[1])))
#  sizes <- unlist(lapply(parsnt,function(el) if(class(el) =="try-error") return(NA) else return(el$pars[2])))
#  sizes <- sapply(sizes,function(x) min(5,x))
#
#  # sd on log scale
#  logsds <- logmeans + log(1+exp(logmeans)/sizes)
#
#  whNA <- which(is.na(logmeans))
#  print(length(whNA))
#  # ?
#  bothmuTs <- cbind(logmeans,muTs[,columnInd]) #estimated log mean for non-tumor and known log mean for tumor
#  bothdispTs <- cbind(sizes,sizeTs[,columnInd]) #estimated size for non-tumor and known size for tumor
#
#  if(length(whNA)> 0){
#    dat <- data[,-whNA]
#    bothmuTs <- bothmuTs[-whNA,]
#    bothdispTs <- bothdispTs[-whNA,]
#  }
#
#  if (columnInd == 1){
#    bothmuTs <- bothmuTs[,c(2,1)]
#    bothdispTs <- bothdispTs[,c(2,1)]
#  }
#
#  ret <- list(mus=bothmuTs, disps=bothdispTs)
#  return(ret)
#
#}
#
#
#estimateTF <- function(data0, muTs, sizeTs) {
#
#  funknown <- 1:dim(data0)[1]
#
#  fests <- estfPGF(dat = t(data0), is = funknown, muTs = muTs, dispTs = sizeTs,
#                   ngrid=25, sum2one=T, upper=1, distr= "nb")
#
#
#  # Retrieve fractions
#  fsremain <- unlist(lapply(fests, function(el) return(1/(1+exp(-el$par)))))
#
#  estimatedTF <- 1-fsremain
#  return(estimatedTF)
#}


getDispersions <- function(dat, priorDF){
    d <- DGEList(counts = dat) #using edgeR
    d <- estimateDisp(d)

    return(list(common=d$common.dispersion, tagwise=d$tagwise.dispersion, trended=d$trended.dispersion))

}



estimateMeanVariance <- function(dataRef, label) {
  # estimate mean and variance of HBDs and tumors
  fssingle <- matrix(0, nrow=dim(dataRef)[1], 2)
  fssingle[label==0,1] = 1
  fssingle[label==1,2] = 1

  fssingle <- t(fssingle)
  dataRef <- t(dataRef)

  nT <- nrow(fssingle)
  nsingle <- ncol(fssingle)

  datsingle <- round(dataRef,0) #data should be integers

  allmus <- alldisps <- c()
  for(t in 1:nT){
    #t<-1
    whin <- which(fssingle[t,]==1)
    dat <- datsingle[,whin]
    mus <- apply(dat,1,mean)
    allmus <- rbind(allmus,mus)
    d <- DGEList(counts = dat) #using edgeR
    d.CR <- estimateCommonDisp(d)
    #if(nrow(dat) >= 100) d.CR <- estimateTrendedDisp(d.CR)
    d.CR <- estimateTagwiseDisp(d.CR)
    disps <- d.CR$tagwise.dispersion
    #var2 <- mus + mus^2*disps
    alldisps <- rbind(alldisps,disps)
  }
  muTs <- t(log(allmus+0.5))
  sizeTs <- t(alldisps)

  return(list(mus=muTs,disps=sizeTs))
}


dmrCallEdgeR <- function(counts, librarySize, label, maxGenes, minFC) {
  # counts in sklearn format, samples x features


  # put data into edgeR, normalize and do differential expression
  y <- DGEList(counts=t(counts), lib.size=librarySize, group=label)
  y <- calcNormFactors(y)

  # print(sort(y[[2]]$norm.factors))

  design <- model.matrix(~label)

  y <- estimateDisp(y,design)

  fit <- glmQLFit(y,design)
  # plotQLDisp(fit)


  dmr <- glmQLFTest(fit, coef=2)

  dmrInfo <- as.data.frame(topTags(dmr, n=dim(counts)[2], adjust.method = 'bonferroni', p.value = 0.05)$table)
  dmrInfo <- dmrInfo[abs(dmrInfo$logFC) > minFC,]

  if (dim(dmrInfo)[1] > maxGenes) {
    dmrInfo <- dmrInfo[1:maxGenes,]
  }

  return(dmrInfo)
}


dmrCallEdgeRstandard <- function(counts, librarySize, label, removeLowCounts, covariates) {
  # counts in sklearn format, samples x features
  # library size = Used reads
  # label 0 1, only binary
  # removeLowCounts: bool whether to call filterByExpr
  # covariates: things to include in the samples argument of DGEList

  print(paste0('Starting DMR calling, ', dim(counts)[1], ' samples, ', dim(counts)[2], ' features'))


  # put data into edgeR, normalize and do differential expression
  if (!is.null(dim(covariates))){
    useCovariates = TRUE
    print(paste('Including', dim(covariates)[2], 'covariates'))
    y <- DGEList(counts=t(counts), lib.size=librarySize, group=label, samples=covariates)

  } else {
    useCovariates = FALSE
    print('No or invalid covariates specified. Using no covariates')
    y <- DGEList(counts=t(counts), lib.size=librarySize, group=label)

  }


  if (removeLowCounts) {
    keep <- filterByExpr(y)
    y <- y[keep, , keep.lib.sizes=TRUE]
    print(paste('Removing regions with low counts.', sum(keep), 'regions remaining.'))
  }

  y <- calcNormFactors(y)

  # print(sort(y[[2]]$norm.factors))
  if (useCovariates) {
    design <- model.matrix(as.formula(paste("~ y$samples$group + ", paste0('y$samples$',colnames(y$samples[,-c(1,2,3),drop=FALSE]), collapse=" + "))))

  } else {
    design <- model.matrix(~label)
  }

  #print(dim(design))
  print(colnames(design))
  y <- estimateDisp(y,design)

  fit <- glmQLFit(y,design)
  # plotQLDisp(fit)


  dmr <- glmQLFTest(fit, coef=2)

  print(paste(dim(topTags(dmr, n=dim(counts)[2], adjust.method = "none", p.value=0.05))[1], "significant before correction"))

  dmrInfo <- as.data.frame(topTags(dmr, n=dim(counts)[2], adjust.method = 'fdr', p.value = 0.05)$table)
  print(paste(dim(dmrInfo)[1], "significant after correction"))
  print("\n\n")


  return(dmrInfo)
}
