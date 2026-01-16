library(rhdf5)

SDMrecovery_file <- "output/SDMrecovery.h5"

# List contents of the h5 file
h5ls(SDMrecovery_file)
sdm_ko <- h5read(SDMrecovery_file, "/")
str(sdm_ko, 2)

# extract the pieces
simulated_data <- sdm_ko$Data$simulatedData
log_posterior <- sdm_ko$Posterior$logposteriors
true_log_params <- sdm_ko$True$TrueLogParms
true_params <- exp(true_log_params)

# inspect
str(simulated_data)
table(simulated_data$id)

# generating parameters
head(true_params)
par(mfrow = c(1, 2))
hist(true_params[, 1], breaks = 40, xlab = "c")
hist(true_params[, 2], breaks = 40, xlab = "kappa")

str(log_posterior)

# extract c est

log_posterior$`r_id__c[1,Intercept]`
subj_c_cols <- paste0("r_id__c[", 1:200, ",Intercept]")

subj_est_c <- log_posterior[, subj_c_cols] + log_posterior$b_c_Intercept
subj_c_medians <- apply(subj_est_c, 2, median)
subj_c_means <- apply(subj_est_c, 2, mean)


subj_k_cols <- paste0("r_id__kappa[", 1:200, ",Intercept]")

subj_est_k <- log_posterior[, subj_k_cols] + log_posterior$b_kappa_Intercept
subj_k_medians <- apply(subj_est_k, 2, median)
subj_k_means <- apply(subj_est_k, 2, mean)


hist(subj_c_medians)
hist(subj_c_means)

plot(true_log_params[, 1], subj_c_means)
abline(0, 1)
plot(exp(true_log_params[, 1]), exp(subj_c_means))
abline(0, 1)

cor(true_log_params[, 1], subj_c_means)
abline(0, 1)
cor(exp(true_log_params[, 1]), exp(subj_c_means))

hist(subj_k_medians)
hist(subj_k_means)


c2 <- bmm::c_sqrtexp2bessel(exp(subj_c_means), exp(subj_k_means))

plot(exp(subj_c_means), c2)

plot(exp(true_log_params[, 1]), c2)
abline(0, 1)


plot(exp(true_log_params[, 2]), exp(subj_k_means))
abline(0, 1)
cor(exp(true_log_params[, 2]), exp(subj_k_means))
