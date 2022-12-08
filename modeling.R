rm(list = objects())

# load libs and sources
if (!require("randtoolbox")) {
  install.packages("randtoolbox")
  library("randtoolbox")
}
if (!require("doParallel")) {
  install.packages("doParallel")
  library("doParallel")
}
if (!require("foreach")) {
  install.packages("foreach")
  library("foreach")
}

# source model and other functions
source("rl_functions.R")

# set seed for reproducibility (the grid search algorithm uses random values)
set.seed(12345)

# load data
df <- read.csv("bandit_data_no_na.csv")


# define parameter estimation function
estimate_parameters <- function(participant_data) {
  # get list of ids
  id_list <- unique(participant_data$id)
  
  # set the maximum number of iterations for each initial optim run:
  startIter <- 20
  
  # set the number of values for which to run optim in full
  fullIter <- 1000
  
  # define parameters and their boundaries (alpha_reward, alpha_skill, w, beta)
  lower_ceilings <- rep(0.00001, 4)
  upper_ceilings <- c(.99999, 1, 1, 200)
  starting_params <- generate_starting_values(200 , min = lower_ceilings, max = upper_ceilings)
  
  # prepare parallel computing
  cores <- detectCores()
  cl <- makeCluster(cores[1] - 1)  # -1 to avoid crashing any PCs
  registerDoParallel(cl)
  
  # estimate parameters
  results <- foreach(i = 1:length(id_list), .combine = rbind, .packages = c("randtoolbox"), .export=ls(.GlobalEnv)) %dopar% {
    temp_data <- c()  # initialize temporary data storage
    d <- subset(df, id == id_list[i])
    
    # find best parameters
    opt <- apply(starting_params, 1, function(x) optim(x, fn = neg_lik, part_data = d, control=list(maxit = startIter), 
                                                       lower = lower_ceilings, upper = upper_ceilings, method = "L-BFGS-B"))
    starting_values_2 <- lapply(opt[order(unlist(lapply(opt,function(x) x$value)))[1:5]],function(x) x$par)
    opt <- lapply(starting_values_2, optim, fn = neg_lik, part_data = d, lower = lower_ceilings, upper = upper_ceilings, method = "L-BFGS-B")
    bestopt <- opt[[which.min(unlist(lapply(opt,function(x) x$value)))]]
    
    # store in results
    temp_data[1] <- id_list[i]  # save id
    temp_data[2] <- bestopt$par[1]  # save alpha_rewards
    temp_data[3] <- bestopt$par[2]  # save alpha_skill
    temp_data[4] <- bestopt$par[3]  # save w
    temp_data[5] <- bestopt$par[4]  # save beta
    temp_data[6] <- bestopt$convergence  # save convergence
    
    temp_data
  }
  
  # stop the cluster
  stopCluster(cl)
  
  colnames(results) <- c("id", "alpha_reward", "alpha_skill", "w", "beta", "convergence")
  
  return(as.data.frame(results))
}

# run estimation
results <- estimate_parameters(df)
results
# view convergences
table(results$convergence)

# remove results that didn't converge
converged_results <- subset(results, convergence == 0)

write.csv(converged_results, file = "modeling_results_new.csv")