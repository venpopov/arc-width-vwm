# Parameter Recovery for hierarchical SDM Measurement Model using categorized errors and aggregated frequency data
# Original script by Klaus Oberauer
rm(list = ls())
graphics.off()
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) # sets the directory of this script as the current directory

# install.packages("BiocManager")
# BiocManager::install("rhdf5", force=T)

library(coda)
library(circular)
library(parallel)
library(binhf) # for shift function: circular shift of a vector
library(Hmisc)
library(cmdstanr)
library(tidybayes)
library(brms)
library(bmm)
library(rhdf5)


# computer <- 2 # 1 = mlsim-server, 2 = laptop, 3 = VM
#
# if (computer == 1) {
#   source("C:/mlsim/R/toolbox/circfun.R")
#   source("C:/mlsim/R/toolbox/HDIofMCMC.R")
#   source("C:/mlsim/R/toolbox/plotPostKO.R")
#   subfolder <- "./Categ/"
# }
# if (computer == 2) {
#   source("C:/Daten/R/toolbox/circfun.R")
#   source("C:/Daten/R/Bayes/toolbox/HDIofMCMC.R")
#   source("C:/Daten/R/Bayes/toolbox/plotPostKO.R")
#   source("C:/Daten/R/Projects/VisWM/SDM/SDMdprimeLogInd.R")
#   subfolder <- "./Categ/"
# }
# if (computer == 3) {
#   source("circfun.R")
#   subfolder <- ""
# }
#
# if (computer == 1) set_cmdstan_path(path = "C:/users/kobera/.cmdstan/cmdstan-2.36.0")
# if (computer == 2) set_cmdstan_path(path = "C:/Users/kobera/.cmdstan/cmdstan-2.35.0")


### Decisions

doFit <- T
verbose <- T
ncores <- parallel::detectCores()
nburnin <- 1000
niterations <- 2000 # after burnin, per chain
nchains <- min(4, ncores - 1) # must not exceed number of cores on the machine!
thinning <- 5
MCMCfile <- "SDM.bmm.recovery.RData"

setsize <- 5
nSubj <- 200 # 200
nTrials <- 100

meanC <- 4
meanKappa <- 2
varC <- 1.5^2
varKappa <- (meanKappa / 5)^2
trueC <- rgamma(nSubj, meanC^2 / varC, meanC / varC)
trueKappa <- rgamma(nSubj, meanKappa^2 / varKappa, meanKappa / varKappa)
TrueLogParms <- matrix(0, nSubj, 2)
TrueLogParms[, 1] <- log(trueC)
TrueLogParms[, 2] <- log(trueKappa)

######### Start Running Code ###################

parNames.bmm <- c("b_c_Intercept", "b_kappa_Intercept")
parnames.bmm <- c("c", "kappa")

### Prepare variables for model fit

postK.bmm <- matrix(0, 1, 3)
postC.bmm <- matrix(0, 1, 3)


### Simulate data

if (doFit) {
  Dataframe <- as.data.frame(matrix(NA, nSubj * nTrials, 3 + setsize))
  names(Dataframe) <- c("id", paste0("Feature", 1:setsize), "response", "error")
  Dataframe[, 2:(setsize + 1)] <- round(runif(nSubj * nTrials * setsize, 1, 360)) # memory features
  Location <- matrix(deg2rad(round(runif(nSubj * nTrials * setsize, 0, 360))), nSubj * nTrials, setsize) # distances from the target
  Distance <- abs(wrap(Location - rep(Location[, 1], setsize)))
  idx <- 1

  ### Simulate data

  for (subj in 1:nSubj) {
    data <- list(
      N = nTrials,
      setsize = setsize,
      m = Dataframe[idx:(idx + nTrials - 1), 2:(setsize + 1)],
      D = Distance,
      response = NA
    )

    pred <- SDMdprimeLogInd(TrueLogParms[subj, ], data, c(2, 4), whatReturn = 2)
    response <- rep(0, nTrials)
    for (trial in 1:nTrials) response[trial] <- sample(x = 1:360, size = 1, prob = pred[trial, ])
    error <- abs(wrap(Dataframe[idx:(idx + nTrials - 1), "Feature1"] - response, 180))
    Dataframe[idx:(idx + nTrials - 1), "id"] <- subj
    Dataframe[idx:(idx + nTrials - 1), "response"] <- deg2rad(response)
    Dataframe[idx:(idx + nTrials - 1), "error"] <- deg2rad(error)
    idx <- idx + nTrials
  }

  #### fit individual trials with bmm

  # define formula
  ff <- bmf(
    c ~ 1 + (1 | p1 | id),
    kappa ~ 1 + (1 | p1 | id)
  )

  # Ven on |p1|: By default, brms treats each formula independently and estimates only the covariance structure within it.
  # By writing |<arbitrary variable>|, it also estimates covariances between parameters of different formulas (here, between c and kappa)

  default_prior(ff,
    data = Dataframe,
    model = sdm(resp_error = "error")
  )

  # set a narrower prior on kappa (though this does not appear to help with recovery)
  newPrior <- set_prior("student_t(5, 1.75, 0.35)", class = "Intercept", dpar = "kappa") +
    set_prior("student_t(1, 0, 0.5)", class = "sd", dpar = "kappa")


  # fit the model
  fit_bmm <- bmm(
    formula = ff,
    data = Dataframe,
    model = sdm(resp_error = "error"),
    prior = newPrior,
    cores = 4,
    iter = niterations,
    init = 1,
    backend = "cmdstanr",
    seed = 123
  )

  # extract the posterior draws
  logposteriors <- as.data.frame(tidybayes::tidy_draws(fit_bmm))

  save(TrueLogParms, logposteriors, file = MCMCfile)
} else {
  load(MCMCfile)
}


################ Analyze ########################

### population parameter estimates from bmm

posteriors <- exp(logposteriors) # all parameters are on the log scale!
postK.bmm[1, 1] <- mean(posteriors[, "b_kappa_Intercept"])
postK.bmm[1, 2:3] <- HDIofMCMC(posteriors[, "b_kappa_Intercept"])
postC.bmm[1, 1] <- mean(posteriors[, "b_c_Intercept"])
postC.bmm[1, 2:3] <- HDIofMCMC(posteriors[, "b_c_Intercept"])

### individual parameter estimates from SDM-cat

indPar.bmm <- matrix(NA, nSubj, 4)
colnames(indPar.bmm) <- c("True Parameter", "Estimated Parameter (Mean)", "HDI-upper", "HDI-lower")

x11(height = 10, width = 10)
layout(matrix(1:4, 2, 2))
for (par in 1:2) {
  truePar <- exp(TrueLogParms[, par])
  sortOrder <- order(truePar)
  indPar.bmm[, 1] <- truePar[sortOrder]
  estPar.bmm <- matrix(NA, length(truePar), 3)
  for (subj in 1:length(truePar)) {
    meanParName <- paste0("b_", parnames.bmm[par], "_Intercept")
    subjParName <- paste0("r_id__", parnames.bmm[par], "[", subj, ",Intercept]")
    postPar <- exp(logposteriors[, meanParName] + logposteriors[, subjParName])
    estPar.bmm[subj, 1] <- mean(postPar)
    estPar.bmm[subj, 2:3] <- HDIofMCMC(postPar)
  }
  indPar.bmm[, 2] <- estPar.bmm[sortOrder, 1]
  indPar.bmm[, 3] <- estPar.bmm[sortOrder, 2]
  indPar.bmm[, 4] <- estPar.bmm[sortOrder, 3]

  ylim <- c(0, max(max(truePar), max(estPar.bmm[, 1])))
  plot(1:nSubj, indPar.bmm[, 1], type = "p", xlab = "Subject", ylab = "Parameter Value", ylim = ylim, main = paste0("bmm: ", parnames.bmm[par]))
  errbar(x = 1:nSubj, y = indPar.bmm[, 2], yplus = indPar.bmm[, 3], yminus = indPar.bmm[, 4], add = T, errbar.col = "red", xlab = "", ylab = "", ylim = ylim)

  dc <- data.frame(indPar.bmm[, 1], indPar.bmm[, 2])
  cc <- cor(dc)
  plot(indPar.bmm[, 1], indPar.bmm[, 2], type = "p", xlab = "True Parameter", ylab = "Estimated Parameter", xlim = ylim, ylim = ylim, main = paste0("bmm: ", parnames.bmm[par]))
  errbar(x = indPar.bmm[, 1], y = indPar.bmm[, 2], yplus = indPar.bmm[, 3], yminus = indPar.bmm[, 4], add = T, errbar.col = "black", xlab = "", ylab = "", ylim = ylim)
  abline(0, 1, col = "red")
  text(0.2 * max(ylim), max(ylim) - 0.5, paste0("r = ", round(cc[2, 1], 2)))
}


### Save in HDF5 format

h5createFile("SDMrecovery.h5")
# creates the file with the desired name in the current directory

h5createGroup("SDMrecovery.h5", "Data")
h5write(Dataframe, file = "SDMrecovery.h5", name = "Data/simulatedData")
h5createGroup("SDMrecovery.h5", "True")
h5write(TrueLogParms, file = "SDMrecovery.h5", name = "True/TrueLogParms")
h5createGroup("SDMrecovery.h5", "Posterior")
h5write(logposteriors, file = "SDMrecovery.h5", name = "Posterior/logposteriors")

h5closeAll()
