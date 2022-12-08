# learning function ----
comp_model <- function(data, n_options, alpha_reward, alpha_skill, w, starting_mean_reward, starting_mean_skill) {
  n_choices <- length(data$choice)
  reward <- data$reward
  skill <- data$ratio
  choices <- data$choice
  reward_estimates <- matrix(starting_mean_reward, ncol = n_options, nrow = n_choices + 1)
  skill_estimates <- matrix(starting_mean_skill, ncol = n_options, nrow = n_choices + 1)
  expected_values <- matrix(starting_mean_reward * starting_mean_skill, ncol = n_options, nrow = n_choices + 1)
  overall_values <- matrix(w * expected_values + (1 - w) * reward_estimates, ncol = n_options, nrow = n_choices + 1)
  
  for (i in 1:n_choices) {
    choice <- rep(0, n_options)
    choice[choices[i]] <- 1
    reward_estimates[i + 1, ] <- reward_estimates[i, ] + choice * (alpha_reward * (reward[i] - reward_estimates[i, ]))
    skill_estimates[i + 1, ] <- skill_estimates[i, ] + choice * (alpha_skill * (skill[i] - skill_estimates[i, ]))
    expected_values[i + 1, ] <- reward_estimates[i + 1, ] * skill_estimates[i + 1, ]
    overall_values[i + 1, ] <- w * expected_values[i + 1, ] + (1 - w) * reward_estimates[i + 1, ]
  }
  
  return(overall_values)
}

# softmax function ----
# softmax function with only 2 choices
softmax_choice_prob <- function(m, beta, conditions) {
  condition_agent_mapper <- list("1x2" = c(1, 2), "1x3" = c(1, 3), "1x4" = c(1, 4), "2x3" = c(2, 3), 
                                 "2x4" = c(2, 4), "3x4" = c(3, 4))
  actual_m <- matrix(NA, nrow = length(conditions), ncol = 2)
  for (i in 1:length(conditions)) {
    actual_m[i, ] <- m[i, condition_agent_mapper[[conditions[i]]]]
  }
  prob <- exp(beta*actual_m)
  prob <- prob/rowSums(prob) # normalize
  return(prob)
}

# ML function ----
neg_lik <- function(par, part_data) {
  starting_alpha_reward <- par[1]
  starting_alpha_skill <- par[2]
  starting_w <- par[3]
  starting_beta <- par[4]
  conditions <- part_data$condition
  options_number <- 4
  mean_reward <- 4
  mean_skill <- .5
  expectations <- comp_model(data = part_data, alpha_reward = starting_alpha_reward, alpha_skill = starting_alpha_skill, w = starting_w, 
                             n_options = options_number, starting_mean_reward = mean_reward, starting_mean_skill = mean_skill)
  p <- softmax_choice_prob(m = expectations, beta = starting_beta, conditions = conditions)
  choices <- part_data$old_agent_choice
  choice_probs <- p[cbind(1:nrow(part_data), choices)]
  n_like <- -sum(log(choice_probs))
  if(is.na(n_like) | n_like == Inf) n_like <- 1e+300
  return(n_like)
}

# starting value generation function
generate_starting_values <- function(n, min, max) {
  require(randtoolbox)
  if(length(min) != length(max)) stop("min and max should have the same length")
  dim <- length(min)
  # generate Sobol values
  start <- sobol(n, dim = dim)
  # transform these to lie between min and max on each dimension
  for(i in 1:ncol(start)) {
    start[,i] <- min[i] + (max[i] - min[i]) * start[, i]
  }
  return(start)
}