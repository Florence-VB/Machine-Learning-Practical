list.of.packages <- c("nnls", "quadprog", "SuperLearner", "ggplot2", "raster", "sp", "rgdal", "rgeos", "glmnet", "Matrix", "foreach", "KernelKnn", "randomForest")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))
# My working directory
setwd("I://fbs//lab1_satellite")

# train
nightlightstif <- raster("train/nightlights.tif")
clusters <- readOGR("train/shapefiles", "clusters", verbose = FALSE) #remove final / if does not work
countryoutline <- readOGR("train/shapefiles", "countryoutline", verbose = FALSE) #remove final / if does not work

# test
nightlightstif_test <- raster("test/nightlights.tif")
clusters_test <- readOGR("test/shapefiles", "clusters", verbose = FALSE) #remove final / if does not work
countryoutline_test <- readOGR("test/shapefiles", "countryoutline", verbose = FALSE) #remove final / if does not work

#plot of the night snapshos with the country and cluster layouts
#here on the traind data so nigeria
plot(nightlightstif)
plot(countryoutline, bg="transparent", add=TRUE)
plot(clusters, bg="transparent", add=TRUE)


#does the same on Tanzania
plot(nightlightstif_test)
plot(countryoutline_test, bg="transparent", add=TRUE)
plot(clusters_test, bg="transparent", add=TRUE)

### Zoom in to a block
temp <- crop(nightlightstif, clusters[clusters@data$ID==220,]) 
plot(temp)


#computes the mean or reads the mean of intensity of light, slight different size of clusters.
#computes the mean for each surveyed block

# nightlights_get <- c()
# for (i in 1:length(clusters@data$ID)){
#   nig_nightlights_crop <- crop(nightlightstif, clusters[clusters@data$ID==i,])
#   nightlights_get <- c(nightlights_get, mean(nig_nightlights_crop@data@values))
#   print(i)
# }
# saveRDS(nightlights_get, "train/nightlights.rds")
nightlights <- readRDS("train/nightlights.rds")

# nightlights_get <- c()
# for (i in 1:length(clusters_test@data$ID)){
#   nig_nightlights_crop <- crop(nightlightstif_test, clusters_test[clusters_test@data$ID==i,])
#   nightlights_get <- c(nightlights_get, mean(nig_nightlights_crop@data@values))
#   print(i)
# }
# saveRDS(nightlights_get, "test/nightlights.rds")
nightlights_test <- readRDS("test/nightlights.rds")

#reads the consumption data
consumptions <- readRDS("train/consumptions.rds")
consumptions_test <- readRDS("test/consumptions.rds")

#plot the link between nightlights intensity and consumptions for each country
#trained data with T
ggplot(data.frame(consumptions, nightlights), aes(nightlights, consumptions)) + geom_point() + geom_smooth(method = "loess") + ggtitle("Tanzania")

#tested with Nigeria
ggplot(data.frame(consumptions_test, nightlights_test), aes(nightlights_test, consumptions_test)) + geom_point() + geom_smooth(method = "loess")+ggtitle("Nigeria")



######################################################################################################
######################################################################################################
##predicting the nightlights by characteristics of the env on google maps
feats <- readRDS("train/features.rds")
feats_test <- readRDS("test/features.rds")

#split between trainingm and test
training <- data.frame(nightlights, feats)
testing <- data.frame(nightlights=nightlights_test, feats_test)
#all ml techniques in superlearner
listWrappers()

#linear reg to predict consumption with env reatures based on the training data. MSE about 0.37 with 5 cV
set.seed(123)
sl_lm = SuperLearner(Y = consumptions, 
                     X = training, 
                     family = gaussian(), 
                     SL.library = "SL.lm", 
                     cvControl = list(V=5))
sl_lm

sl_lm$fit
sl_lm$coefficients
#Library$SLlm_All$object$coefficients

#ensemble
sl_2 = SuperLearner(Y = consumptions, 
                    X = training, family = gaussian(), SL.library = c("SL.lm", "SL.kernelKnn"), 
                    cvControl = list(V=5))
sl_2



cv_sl_2 = CV.SuperLearner(Y = consumptions, 
                          X = training, family = gaussian(), SL.library = c("SL.lm", "SL.kernelKnn"), 
                          cvControl = list(V=5))
summary(cv_sl_2)
plot(cv_sl_2)



#lasso vs ridge
ridge = create.Learner("SL.glmnet", params = list(alpha = 0), name_prefix="ridge")
lasso = create.Learner("SL.glmnet", params = list(alpha = 1), name_prefix="lasso")

sl_libraries <- c(lasso$names, ridge$names)
sl_en <- SuperLearner(Y = consumptions,
                      X = training, 
                      family = gaussian(),
                      SL.library = sl_libraries, 
                      cvControl = list(V=5))
sl_en

#random forest
mtry = create.Learner("SL.randomForest", tune = list(mtry = c(2,3)), name_prefix="mtry")
sl_libraries <- mtry$names
sl_mtry <- SuperLearner(Y = consumptions,
                        X = training, 
                        family = gaussian(),
                        SL.library = sl_libraries, 
                        cvControl = list(V=5))
sl_mtry



#screening
SL.library=list(c("SL.pred_1", "screen.scr_1"), c("SL.pred_2", "screen.scr_2"), "SL.pred_3")
#compares the RF with lasso screening, lasso and ridge
sl_libraries <- list(c("SL.randomForest", "screen.glmnet"), lasso$names, ridge$names)
sl_screen <- SuperLearner(Y = consumptions,
                          X = training, family = gaussian(),
                          SL.library = sl_libraries, cvControl = list(V=5))
sl_screen


##explores which  ML combination to improve the risk
listWrappers()
sl_libraries <- list(c("SL.randomForest", "screen.glmnet"), c("SL.randomForest", "screen.corP"), c("SL.randomForest", "screen.SIS"))
sl_screen <- SuperLearner(Y = consumptions,
                          X = training, family = gaussian(),
                          SL.library = sl_libraries, cvControl = list(V=5))
sl_screen


#plot the best results based on the SuperLearner model
pred <- predict(sl_screen, testing, onlySL = T)
#plots the predicted consumption values vs the actual consumption values
ggplot(data.frame(consumptions_test, pred$pred), aes(pred$pred , consumptions_test)) + geom_point() + geom_abline(slope=1, intercept=0)
mse = mean((consumptions_test-pred$pred)^2)
print(paste("Mean squared error:", round(mse, digits=4)))
print(paste("Correlation:", round(cor(consumptions_test,pred$pred), digits=4)))


#re update r with the installr package
updateR()