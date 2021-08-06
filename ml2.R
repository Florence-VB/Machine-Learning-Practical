#########################################################################################################################
###generates a random dataset in which we know the true beta value to compare the biases coming from different learners
##########################################################################################################################

generate_data <- function(N=500, k=50, true_beta=1) {
  # DGP inspired by https://www.r-bloggers.com/cross-fitting-double-machine-learning-estimator/ 
  # Generates a list of 3 elements, y, x and w, where y and x are N 
  # times 1 vectors, and w is an N times k matrix. 
  #
  # Args:
  #   N: Number of observations
  #   k: Number of variables in w
  #   true_beta: True value of beta
  #
  # Returns:
  #   a list of 3 elements, y, x and w, where y and x are N 
  #   times 1 vectors, and w is an N times k matrix. 
  
  b=1/(1:k)
  
  # = Generate covariance matrix of w = #
  sigma=genPositiveDefMat(k,"unifcorrmat")$Sigma
  sigma=cov2cor(sigma)
  
  w=rmvnorm(N,sigma=sigma) # Generate w
  g=as.vector(cos(w%*%b)^2) # Generate the function g 
  m=as.vector(sin(w%*%b)+cos(w%*%b)) # Generate the function m 
  x=m+rnorm(N) # Generate x
  y=true_beta*x+g+rnorm(N) # Generate y
  
  dgp = list(y=y, x=x, w=w)
  
  return(dgp)
}

set.seed(123)
dgp <- generate_data()

#############################################################################
####Learners generation
##############################################################################
#####creates the different learners to compare performances
ridge = create.Learner("SL.glmnet", params = list(alpha = 0), name_prefix="ridge")
lasso = create.Learner("SL.glmnet", params = list(alpha = 1), name_prefix="lasso")
en05 = create.Learner("SL.glmnet", params = list(alpha = 0.5), name_prefix="en05")


####################################################################################################################
####OLS VS LASSO
#######################################################################################################################
SL.library <- c(lasso$names, ridge$names, en05$names, "SL.lm")
sl_en <- SuperLearner(Y = dgp$y,
                      X = data.frame(x=dgp$x, w=dgp$w), 
                      family = gaussian(),
                      SL.library = SL.library, 
                      cvControl = list(V=5))
sl_en

#extracts coeff from OLS and the X coeff reflects the predicted beta to be compared to other models
coef(sl_en$fitLibrary$SL.lm_All$object)[1:20] # first 20 OLS coeffs
#stores the lasso coeff
lasso_coeffs <- data.frame(lambda = sl_en$fitLibrary$lasso_1_All$object$lambda)
#keeps only non null
lasso_coeffs$nzero <- sl_en$fitLibrary$lasso_1_All$object$nzero
#MSE with cross validation methods
lasso_coeffs$cvm <- sl_en$fitLibrary$lasso_1_All$object$cvm 

#stores the coeff from minimized lasso MSE
optimal_cv = which.min(lasso_coeffs$cvm)
#gives the min MSE based on CV equals to 5
lasso_coeffs$cvm[optimal_cv]
optimal_cv<-as.data.frame(optimal_cv)

#simulations of linear reg to estimate the avg beta and its bias towards a specific val
beta_X = c()
for (i in 1:100) {
  print(i)
  dgp = generate_data()
  
  SL.library <- "SL.lm"
  sl_lm <- SuperLearner(Y = dgp$y,
                        X = data.frame(x=dgp$x, w=dgp$w), 
                        family = gaussian(),
                        SL.library = SL.library, 
                        cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_lm$fitLibrary$SL.lm_All$object)[2]) #extracts the predicted betas
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X_df$beta_X))



###simulates 100 times the lasso prediction
beta_X = c()
for (i in 1:100) {
  print(i)
  dgp = generate_data()
  
  SL.library <- c(lasso$names)
  sl_lasso <- SuperLearner(Y = dgp$y,
                           X = data.frame(x=dgp$x, w=dgp$w), 
                           family = gaussian(),
                           SL.library = SL.library, 
                           cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_lasso$fitLibrary$lasso_1_All$object, s="lambda.min")[2])#extracts coeff which min lambda
}

beta_X_df <- data.frame(beta_X=beta_X)

print(mean(beta_X_df$beta_X)) #


###########################################
#####Lasso as screening meth
########################################################
### useful function to get lasso coefficients:
get_lasso_coeffs <- function(sl_lasso){
  optimal_lambda_index <- which.min(sl_lasso$fitLibrary$lasso_1_All$object$cvm)
  return(sl_lasso$fitLibrary$lasso_1_All$object$glmnet.fit$beta[,optimal_lambda_index])
}
beta_X = c()
for (i in 1:100) {
  print(i)
  dgp = generate_data()
  
  SL.library <- lasso$names
  sl_lasso <- SuperLearner(Y = dgp$y,
                           X = data.frame(x=dgp$x, w=dgp$w), 
                           family = gaussian(),
                           SL.library = SL.library, 
                           cvControl = list(V=0))
  
  kept_variables <- which(get_lasso_coeffs(sl_lasso)!=0) - 1 # minus 1 as X is listed
  kept_variables <- kept_variables[kept_variables>0]
  
  sl_pred_x <- SuperLearner(Y = dgp$x,
                            X = data.frame(w=dgp$w), 
                            family = gaussian(),
                            SL.library = lasso$names, cvControl = list(V=0))
  
  kept_variables2 <- which(get_lasso_coeffs(sl_pred_x)!=0) 
  kept_variables2 <- kept_variables2[kept_variables2>0]
  
  sl_screening_lasso <- SuperLearner(Y = dgp$y,
                                     X = data.frame(x = dgp$x, w = dgp$w[, c(kept_variables, kept_variables2)]), 
                                     family = gaussian(),
                                     SL.library = "SL.lm", 
                                     cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_screening_lasso$fitLibrary$SL.lm_All$object)[2])
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X_df$beta_X))


#######################################################################
####naive Frisch Waive
########################################################################
beta_X = c()
for (i in 1:100) {
  print(i)
  dgp = generate_data()
  #predicts X on W
  sl_x = SuperLearner(Y = dgp$x, 
                      X = data.frame(w=dgp$w), # the data used to train the model
                      newX= data.frame(w=dgp$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  
  x_hat <- sl_x$SL.predict
  
  #predicts y on w
  sl_y = SuperLearner(Y = dgp$y, 
                      X = data.frame(w=dgp$w), # the data used to train the model
                      newX= data.frame(w=dgp$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  y_hat <-  sl_y$SL.predict
  #residuals from the predictions
  res_x = dgp$x - x_hat
  res_y = dgp$y - y_hat
  beta = (mean(res_x*res_y))/(mean(res_x**2)) # (coefficient of regression of res_y on res_x)
  beta_X = c(beta_X, beta)
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X_df$beta_X))

##################################################################
#######frish waugh with sample splitting
####################################################################
beta_X = c()
for (i in 1:100) {
  print(i)
  dgp = generate_data()
  
  split <- sample(seq_len(length(dgp$y)), size = ceiling(length(dgp$y)/2))
  
  dgp1 = list(y = dgp$y[split], x = dgp$x[split], w = dgp$w[split,])
  dgp2 = list(y = dgp$y[-split], x = dgp$x[-split], w = dgp$w[-split,])
  
  ## This time we train on one sample and predict using the other sample
  #predicts X on W
  sl_x = SuperLearner(Y = dgp1$x, 
                      X = data.frame(w=dgp1$w), # the data used to train the model
                      newX= data.frame(w=dgp1$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  
  x_hat <- sl_x$SL.predict
  
  #predicts y on w
  sl_y = SuperLearner(Y = dgp2$y, 
                      X = data.frame(w=dgp2$w), # the data used to train the model
                      newX= data.frame(w=dgp2$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  y_hat <-  sl_y$SL.predict
  #residuals from the predictions
  res_x = dgp1$x - x_hat
  res_y = dgp2$y - y_hat
  beta = (mean(res_x*res_y))/(mean(res_x*2)) # (coefficient of regression of res_y on res_x)
  beta_X = c(beta_X, beta)
}
beta_X_df <- data.frame(beta_X=beta_X)
ggplot(beta_X_df, aes(x = beta_X)) + geom_histogram(binwidth = 0.001)
print(mean(beta_X_df$beta_X))

################################################################################################################
######pasted from correction
#################################################################################################################
doubleml <- function(X, W, Y, SL.library.X = "SL.lm",  SL.library.Y = "SL.lm", family.X = gaussian(), family.Y = gaussian()) {
  
  ### STEP 1: split X, W and Y into 2 random sets
  split <- sample(seq_len(length(Y)), size = ceiling(length(Y)/2))
  
  Y1 = Y[split]
  Y2 = Y[-split]
  
  X1 = X[split]
  X2 = X[-split]
  
  W1 = W[split, ]
  W2 = W[-split, ]
  
  ### STEP 2a: use a SuperLearner to train a model for E[X|W] on set 1 and predict X on set 2 using this model. Do the same but training on set 2 and predicting on set 1
  sl_x1 = SuperLearner(Y = X1, 
                       X = data.frame(W1), 
                       newX= data.frame(W2), 
                       family = family.X, 
                       SL.library = SL.library.X, 
                       cvControl = list(V=0))
  x_hat_2 <- sl_x1$SL.predict
  
  sl_x2 = SuperLearner(Y = X2, 
                       X = data.frame(W2), 
                       newX= data.frame(W1), 
                       family = family.X, 
                       SL.library = SL.library.X,
                       cvControl = list(V=0))
  x_hat_1 <- sl_x2$SL.predict
  
  ### STEP 2b: get the residuals X - X_hat on set 2 and on set 1
  res_x_2 <- X2 - x_hat_2
  res_x_1 <- X1 - x_hat_1
  
  ### STEP 3a: use a SuperLearner to train a model for E[Y|W] on set 1 and predict Y on set 2 using this model. Do the same but training on set 2 and predicting on set 1
  sl_y1 = SuperLearner(Y = Y1, 
                       X = data.frame(W1), 
                       newX= data.frame(W2), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
  y_hat_2 <- sl_y1$SL.predict
  sl_y2 = SuperLearner(Y = Y2, 
                       X = data.frame(W2), 
                       newX= data.frame(W1), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
  y_hat_1 <- sl_y2$SL.predict
  
  ### STEP 3b: get the residuals Y - Y_hat on set 2 and on set 1
  res_y_2 <- Y2 - y_hat_2
  res_y_1 <- Y1 - y_hat_1
  
  ### STEP 4: regress (Y - Y_hat) on (X - X_hat) on set 1 and on set 2, and get the coefficients of (X - X_hat)
  beta1 = (mean(res_x_1*res_y_1))/(mean(res_x_1*res_x_1))
  beta2 = (mean(res_x_2*res_y_2))/(mean(res_x_2*res_x_2))
  
  ### STEP 5: take the average of these 2 coefficients from the 2 sets (= beta)
  beta=0.5*(beta1+beta2)
  
  ### STEP 6: compute standard errors (done for you). This is just the usual OLS standard errors in the regression res_y = res_x*beta + eps. 
  psi_stack = c((res_y_1-res_x_1*beta), (res_y_2-res_x_2*beta))
  res_stack = c(res_x_1, res_x_2)
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(Y))
  
  return(c(beta,se))
}
##save under a function before simulation
doubleml(X=dgp$x, W=dgp$w, Y=dgp$y, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost")


##########################################################
##simulations 100 times 
############################################################
beta_X = c()
se_X = c()
for (i in 1:100) {
  tryCatch({
    print(i)
    dgp = generate_data()
    
    DML = doubleml(X=dgp$x, W=dgp$w, Y=dgp$y, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost")
    
    beta_X = c(beta_X, DML[1])
    se_X = c(se_X, DML[2])
    
  }, error = function(e) {
    print(paste("Error for", i))
  })
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(se_X))
print(mean(beta_X))

##################################################################################################################
#################################################################################################################
###################################################################################################################

###part 3
##########
##reads data
db<-read.csv2(file="I://fbs//ML//social_turnout.csv", header=TRUE, sep=",", dec=".")
str(db)
dbW<-db[,c(1:44)]

doubleml <- function(X, W, Y, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X =  binomial(), family.Y = binomial()) {
  
  ### STEP 1: split X, W and Y into 2 random sets
  split <- sample(seq_len(length(db)), size = ceiling(length(db)/2))
  
  Y1 = db$outcome_voted[split]
  Y2 = db$outcome_voted[-split]
  
  X1 = db$treat_neighbors[split]
  X2 = db$treat_neighbors[-split]
  
  W1 = dbW[split, ]
  W2 = dbW[-split, ]
  
  ### STEP 2a: use a SuperLearner to train a model for E[X|W] on set 1 and predict X on set 2 using this model. Do the same but training on set 2 and predicting on set 1
  sl_x1 = SuperLearner(Y = X1, 
                       X = data.frame(W1), 
                       newX= data.frame(W2), 
                       family = family.X, 
                       SL.library = SL.library.X, 
                       cvControl = list(V=0))
  x_hat_2 <- sl_x1$SL.predict
  
  sl_x2 = SuperLearner(Y = X2, 
                       X = data.frame(W2), 
                       newX= data.frame(W1), 
                       family = family.X, 
                       SL.library = SL.library.X,
                       cvControl = list(V=0))
  x_hat_1 <- sl_x2$SL.predict
  
  ### STEP 2b: get the residuals X - X_hat on set 2 and on set 1
  res_x_2 <- X2 - x_hat_2
  res_x_1 <- X1 - x_hat_1
  
  ### STEP 3a: use a SuperLearner to train a model for E[Y|W] on set 1 and predict Y on set 2 using this model. Do the same but training on set 2 and predicting on set 1
  sl_y1 = SuperLearner(Y = Y1, 
                       X = data.frame(W1), 
                       newX= data.frame(W2), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
  y_hat_2 <- sl_y1$SL.predict
  sl_y2 = SuperLearner(Y = Y2, 
                       X = data.frame(W2), 
                       newX= data.frame(W1), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
  y_hat_1 <- sl_y2$SL.predict
  
  ### STEP 3b: get the residuals Y - Y_hat on set 2 and on set 1
  res_y_2 <- Y2 - y_hat_2
  res_y_1 <- Y1 - y_hat_1
  
  ### STEP 4: regress (Y - Y_hat) on (X - X_hat) on set 1 and on set 2, and get the coefficients of (X - X_hat)
  beta1 = (mean(res_x_1*res_y_1))/(mean(res_x_1*res_x_1))
  beta2 = (mean(res_x_2*res_y_2))/(mean(res_x_2*res_x_2))
  
  ### STEP 5: take the average of these 2 coefficients from the 2 sets (= beta)
  beta=0.5*(beta1+beta2)
  
  
  ### STEP 6: compute standard errors (done for you). This is just the usual OLS standard errors in the regression res_y = res_x*beta + eps. 
  psi_stack = c((res_y_1-res_x_1*beta), (res_y_2-res_x_2*beta))
  res_stack = c(res_x_1, res_x_2)
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(Y))
  
  return(c(beta,se))
}
doubleml(X=db$treat_neighbors, W=dbW, Y=db$outcome_voted, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost")

############################################################################
######Check with multiple simulated datasets
beta_X = c()
se_X = c()
for (i in 1:30) {
  tryCatch({
    print(i)
    DML = doubleml(X=db$treat_neighbors, W=dbW, Y=db$outcome_voted, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost")
    
    beta_X = c(beta_X, DML[1])
    se_X = c(se_X, DML[2])
    
  }, error = function(e) {
    print(paste("Error for", i))
  })
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(se_X))
print(mean(beta_X))
##########################################################################################
#####How does OLS behaves with W variables
#simulations of linear reg to estimate the avg beta and its bias towards a specific val
beta_X = c()
for (i in 1:100) {
  print(i)
  #here linear model
  SL.library <- "SL.lm"
  sl_lm <- SuperLearner(Y = db$outcome_voted,
                        X = data.frame(x=db$treat_neighbors, w=dbW), 
                       family = binomial(),
                        SL.library = "SL.xgboost", 
                        cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_lm$fitLibrary$SL.lm_All$object)[2]) #extracts the predicted betas
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X_df$beta_X))

#without W
beta_X = c()
for (i in 1:100) {
  print(i)
  #dgp = generate_data()
  
  SL.library <- "SL.lm"
  sl_lm <- SuperLearner(Y = db$outcome_voted,
                        X = data.frame(x=db$treat_neighbors), 
                        family = binomial(),
                        SL.library = "SL.xgboost", 
                        cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_lm$fitLibrary$SL.lm_All$object)[2]) #extracts the predicted betas
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X_df$beta_X))

####################################################################################################
########################################################################################
###starts with k=4 with two auxilliary samples
db1<-db %>% dplyr::filter(row_number() %% 2 == 0) ## Select even rows
db2<- db %>% dplyr::filter(row_number() %% 2 == 1) ##selects uneven rows
#defines the different outcomes and treatment to estimate
Y1<-db1$outcome_voted
Y2<-db2$outcome_voted
X1<-db1$treat_neighbors
X2<-db2$treat_neighbors
##characteristics to predict x and y
dbW1<-db1[, c(1:44)]
dbW2<-db2[, c(1:44)]
table(db$treat_neighbors)
table(db$outcome_voted)#

beta<-NULL
beta1<-NULL
beta2<-NULL

###double ML with CF
SL.library.X = "SL.xgboost"
SL.library.Y = "SL.xgboost"
family.X =  binomial()
family.Y = binomial()
doubleml_CF <- function(X, W, Y, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X =  binomial(), family.Y = binomial()) {

### SuperLearner to train a model for E[X|W] on set 1 and predict X on set full sample
sl_x1 = SuperLearner(Y = X1, 
                     X = data.frame(dbW1), 
                     newX= data.frame(dbW), 
                     family = family.X, 
                     SL.library = SL.library.X, 
                     cvControl = list(V=0))
x_hat_2 <- sl_x1$SL.predict

### SuperLearner to train a model on all data and predict on db1
sl_x2 = SuperLearner(Y = db$treat_neighbors, 
                     X = data.frame(dbW), 
                     newX= data.frame(dbW1), 
                     family = family.X, 
                     SL.library = SL.library.X, 
                     cvControl = list(V=0))
x_hat_1 <- sl_x2$SL.predict


### SuperLearner to train a model for E[X|W] on set 2 and predict X on set full sample
#sl_x3 = SuperLearner(Y = X2, 
  #                   X = data.frame(dbW2), 
 #                    newX= data.frame(dbW), 
   #                  family = binomial(), 
    #                 SL.library = SL.library.X, 
    #                 cvControl = list(V=0))
#x_hat_4 <- sl_x3$SL.predict

### SuperLearner to train a model on all data and predict on db1
#sl_x4 = SuperLearner(Y = db$treat_neighbors, 
 #                    X = data.frame(dbW), 
  #                   newX= data.frame(dbW1), 
  #                   family = binomial(), 
   #                  SL.library = SL.library.X, 
   #                  cvControl = list(V=0))
#x_hat_3 <- sl_x4$SL.predict



### get the residuals X - X_hat on set 2 and on set 1
res_x_2 <- db$treat_neighbors - x_hat_2
res_x_1 <- X1 - x_hat_1
# <- X2 - x_hat_3
#res_x_4<-db$treat_neighbors - x_hat_4

###  SuperLearner to train a model for E[Y|W] on set 1 and predict on full set
sl_y1 = SuperLearner(Y = Y1, 
                     X = data.frame(dbW1), 
                     newX= data.frame(dbW), 
                     family = family.Y, 
                     SL.library = SL.library.Y ,
                     cvControl = list(V=0))
y_hat_2 <- sl_y1$SL.predict


###  SuperLearner to train a model for E[Y|W] on full set and predict on db1
sl_y2 = SuperLearner(Y = db$outcome_voted, 
                     X = data.frame(dbW), 
                     newX= data.frame(dbW1), 
                     family = family.Y, 
                     SL.library = SL.library.Y , 
                     cvControl = list(V=0))
y_hat_1 <- sl_y2$SL.predict

### SuperLearner to train a model for E[X|W] on set 2 and predict X on set full sample
#sl_y3 <- SuperLearner(Y = Y2, 
  #                   X = data.frame(dbW2), 
 #                    newX= data.frame(dbW), 
                    # family = family.Y, 
  #                   SL.library = SL.library.Y , 
  #                   cvControl = list(V=0))
#y_hat_4 <- sl_y3$SL.predict

### SuperLearner to train a model on all data and predict on db2
# = SuperLearner(Y = db$outcome_voted, 
      #               X = data.frame(dbW), 
       #              newX= data.frame(dbW2), 
      ##               family = family.Y, 
       #              SL.library = SL.library.Y , 
        #             cvControl = list(V=0))
#y_hat_3 <- sl_y4$SL.predict


### STEP 3b: get the residuals Y - Y_hat on set 2 and on set 1
res_y_2 <- db$outcome_voted - y_hat_2
res_y_1 <- Y1 - y_hat_1
#res_y_3<- Y2 - y_hat_3
#res_y_4<- db$outcome_voted - y_hat_4
### STEP 4: regress (Y - Y_hat) on (X - X_hat) on set 1 and on set 2, and get the coefficients of (X - X_hat)
beta1 = (mean(res_x_1*res_y_1))/(mean(res_x_1*res_x_1))
beta2 = (mean(res_x_2*res_y_2))/(mean(res_x_2*res_x_2))
#beta3 = (mean(res_x_3*res_y_3))/(mean(res_x_3*res_x_3))
#beta4 = (mean(res_x_4*res_y_4))/(mean(res_x_4*res_x_4))
### STEP 5: take the average of these 2 coefficients from the 2 sets (= beta)
#beta=0.25*(beta1+beta2+beta3+beta4)
beta<-0.5*(beta1+beta2)

return(c(beta))
}
doubleml_CF(X=db$treat_neighbors, W=dbW, Y=db$outcome_voted)


#runs a 30 simulations on the double ML with CF
#plots to see if non linear distribution
beta_X = c()
for (i in 1:30) {
  tryCatch({
    print(i)
    DML = doubleml_CF(X=db$treat_neighbors, W=dbW, Y=db$outcome_voted, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X = binomial(), family.Y = binomial())
    
    beta_X = c(beta_X, DML[1])

  }, error = function(e) {
    print(paste("Error for", i))
  })
}

beta_X_df <- data.frame(beta_X=beta_X)
print(mean(beta_X))




















##################################
#####attempt with k samples
library(clusterGeneration)
library(mvtnorm)
library(randomForest)

set.seed(123) # = Seed for Replication = #
N=length(db$outcome_voted)

########################to compute avg
set.seed(123)
M=100 # defines 100 simulations
# Matrix to store 
thetahat=matrix(NA,M,1)
colnames(thetahat)=c("Cross-fiting DML")

##loop to computes the DML
###Y=Dtheta + G(X)+U with E[U|D,X]=0
###D=m(X)+V with E[V|X]=0
for(i in 1:M){
  z=rmvnorm(N,sigma=sigma) # = Generate z = #
  g=as.vector(cos(z%*%b)^2) # = Generate the function g = #
  m=as.vector(sin(z%*%b)+cos(z%*%b)) # = Generate the function m = #
  d=m+rnorm(N) # = Generate d = #
  y=theta*d+g+rnorm(N) # = Generate y = 


# = Cross-fitting DML = #
# = Split sample = #
I=sort(sample(1:N,N/2))
IC=setdiff(1:N,I)










k=10 # = Number of variables in z = #
theta=0.5
b=1/(1:k)

# = Generate covariance matrix of z = #
sigma=genPositiveDefMat(k,"unifcorrmat")$Sigma
sigma=cov2cor(sigma)


set.seed(123)
M=500 # = Number of Simumations = #

# = Matrix to store results = #
thetahat=matrix(NA,M,3)
colnames(thetahat)=c("OLS","Naive DML","Cross-fiting DML")

for(i in 1:M){
  z=rmvnorm(N,sigma=sigma) # = Generate z = #
  g=as.vector(cos(z%*%b)^2) # = Generate the function g = #
  m=as.vector(sin(z%*%b)+cos(z%*%b)) # = Generate the function m = #
  d=m+rnorm(N) # = Generate d = #
  y=theta*d+g+rnorm(N) # = Generate y = #
  
  # = OLS estimate = #
  OLS=coef(lm(y~d))[2]
  thetahat[i,1]=OLS
  
  # = Naive DML = #
  # = Compute ghat = #
  model=randomForest(z,y,maxnodes = 20)
  G=predict(model,z)
  # = Compute mhat = #
  modeld=randomForest(z,d,maxnodes = 20)
  M=predict(modeld,z)
  # = compute vhat as the residuals of the second model = #
  V=d-M
  # = Compute DML theta = #
  theta_nv=mean(V*(y-G))/mean(V*d)
  thetahat[i,2]=theta_nv
  
  # = Cross-fitting DML = #
  # = Split sample = #
  I=sort(sample(1:N,N/2))
  IC=setdiff(1:N,I)
  # = compute ghat on both sample = #
  model1=randomForest(z[IC,],y[IC],maxnodes = 10)
  model2=randomForest(z[I,],y[I], maxnodes = 10)
  G1=predict(model1,z[I,])
  G2=predict(model2,z[IC,])
  
  # = Compute mhat and vhat on both samples = #
  modeld1=randomForest(z[IC,],d[IC],maxnodes = 10)
  modeld2=randomForest(z[I,],d[I],maxnodes = 10)
  M1=predict(modeld1,z[I,])
  M2=predict(modeld2,z[IC,])
  V1=d[I]-M1
  V2=d[IC]-M2
  
  # = Compute Cross-Fitting DML theta
  theta1=mean(V1*(y[I]-G1))/mean(V1*d[I])
  theta2=mean(V2*(y[IC]-G2))/mean(V2*d[IC])
  theta_cf=mean(c(theta1,theta2))
  thetahat[i,3]=theta_cf
  
}

colMeans(thetahat) # = check the average theta for all models = #


# = plot distributions = #
plot(density(thetahat[,1]),xlim=c(0.3,0.7),ylim=c(0,14))
lines(density(thetahat[,2]),col=2)
lines(density(thetahat[,3]),col=4)
abline(v=0.5,lty=2,col=3)
legend("topleft",legend=c("OLS","Naive DML","Cross-fiting DML"),col=c(1,2,4),lty=1,cex=0.7,seg.len = 0.7,bty="n")