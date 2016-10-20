library(data.table)
library(ggplot2)
library(dplyr)
library(caret)
train = fread('train.csv',data.table = F)
test = fread('test-comb.csv',data.table = F)
test = test %>% select(-V1,-Comb)





# Gender
train$Gender = as.numeric(as.factor(train$Gender))
test$Gender = as.numeric(as.factor(test$Gender))

#Age:

train$Age = as.numeric(as.factor(train$Age))
test$Age = as.numeric(as.factor(test$Age))


#City:
train$City_Category = as.numeric(as.factor(train$City_Category))
test$City_Category = as.numeric(as.factor(test$City_Category))


#pr_id and usr_id
subm1 =  test %>% select(User_ID,Product_ID)
Purchase = train$Purchase
train = train %>% select(-Purchase)
all = rbind(train,test)
all$User_ID = as.numeric(as.factor(all$User_ID))
all$Product_ID = as.numeric(as.factor(all$Product_ID))

train = all[1:nrow(train),]
train = cbind(Purchase,train)

test = all[(nrow(train)+1):nrow(all),]







#stay year
train$Stay_In_Current_City_Years = as.numeric(as.factor(train$Stay_In_Current_City_Years))
test$Stay_In_Current_City_Years = as.numeric(as.factor(test$Stay_In_Current_City_Years))


#simple imputation
train[is.na(train$Product_Category_2),]$Product_Category_2 = 0
test[is.na(test$Product_Category_2),]$Product_Category_2 = 0

train[is.na(train$Product_Category_3),]$Product_Category_3 = 0
test[is.na(test$Product_Category_3),]$Product_Category_3 = 0

########category features:
train$cat12 = train$Product_Category_1 + train$Product_Category_2
train$cat13 = train$Product_Category_1 + train$Product_Category_3
train$cat23 = train$Product_Category_2 + train$Product_Category_3
train$cat123 = train$Product_Category_1 + train$Product_Category_2 + train$Product_Category_3

test$cat12 = test$Product_Category_1 + test$Product_Category_2
test$cat13 = test$Product_Category_1 + test$Product_Category_3
test$cat23 = test$Product_Category_2 + test$Product_Category_3
test$cat123 = test$Product_Category_1 + test$Product_Category_2 + test$Product_Category_3






#cat pct
# train$pct_c1 = train$Product_Category_1/(train$Product_Category_1+
#                                                  train$Product_Category_2+
#                                                  train$Product_Category_3)
# train$pct_c2 = train$Product_Category_2/(train$Product_Category_1+
#                                                  train$Product_Category_2+
#                                                  train$Product_Category_3)
# train$pct_c3 = train$Product_Category_3/(train$Product_Category_1+
#                                                  train$Product_Category_2+
#                                                  train$Product_Category_3)
#
#
# test$pct_c1 = test$Product_Category_1/(test$Product_Category_1+
#                                                test$Product_Category_2+
#                                                test$Product_Category_3)
# test$pct_c2 = test$Product_Category_2/(test$Product_Category_1+
#                                                test$Product_Category_2+
#                                                test$Product_Category_3)
# test$pct_c3 = test$Product_Category_3/(test$Product_Category_1+
#                                                test$Product_Category_2+
#                                                test$Product_Category_3)




#####squared feature
sq_train = apply(train %>% select(-Purchase),2,function(x){
        x^2
})
colnames(sq_train) = sapply(colnames(sq_train),function(x) paste('sq',x, sep = '_'))
####
sq_test = apply(test,2,function(x){
        x^2
})
colnames(sq_test) = sapply(colnames(sq_test),function(x) paste('sq',x, sep = '_'))


#####third feature
th_train = apply(train %>% select(-Purchase),2,function(x){
        x^3
})
colnames(th_train) = sapply(colnames(th_train),function(x) paste('th',x, sep = '_'))
####
th_test = apply(test,2,function(x){
        x^3
})
colnames(th_test) = sapply(colnames(th_test),function(x) paste('th',x, sep = '_'))


#####log feature
lg_train = apply(train %>% select(-Purchase),2,function(x){
        log(x+1)
})
colnames(lg_train) = sapply(colnames(lg_train),function(x) paste('lg',x, sep = '_'))
####
lg_test = apply(test,2,function(x){
        log(x+1)
})
colnames(lg_test) = sapply(colnames(lg_test),function(x) paste('lg',x, sep = '_'))

#####exponential feature
exp_train = apply(train %>% select(-Purchase),2,function(x){
        exp(x)
})
colnames(exp_train) = sapply(colnames(exp_train),function(x) paste('exp',x, sep = '_'))
####
exp_test = apply(test,2,function(x){
        exp(x)
})
colnames(exp_test) = sapply(colnames(exp_test),function(x) paste('exp',x, sep = '_'))
################
################
#### interaction vars:
label=train$Purchase
train2=as.data.frame(model.matrix( Purchase~ .^3,train))
train=cbind(train2,label)
rm(train2)
test=as.data.frame(model.matrix( ~ .^3,test))
##########################
train = cbind(sq_train,train)
test = cbind(sq_test,test)
train = cbind(th_train,train)
test = cbind(th_test,test)
train = cbind(lg_train,train)
test = cbind(lg_test,test)
train = cbind(exp_train,train)
test = cbind(exp_test,test)

rm(exp_test,exp_train,lg_train,lg_test,sq_train,sq_test,
   th_train,th_test,label)




######################
label = train$label
train = train %>% select(-label)
######################

##########################




# train = as.data.frame(train)
# train=cbind(train,label)
# test=as.data.frame(test)
########## Using xgboost gbtree:
#10:              cv:2848.99     lb:2890    (using top 71 features)
#11:              cv:2837.6      lb:2883    (using top 45 features)
#12:              cv:2836.6      lb:2879   (using top 40 features)
#13:              cv:28  lb:       (using top 35 features)
########## Using xgboost gbtree w/ extra features
#14:              cv:2852      lb:2853   (using top 40 features)
#15:              cv:2853.85   lb:2853  (using top 35 features)
##########Eta=0.3
#16:              cv:2851.406      lb:2851.98  (using top 40 features)
##############use prod_id and user_id
#17:        cv: 2505      lb:2512  (using top 40 features)
#17:        cv: 2506      lb:2507  (using top 45 features)





###
#Next Step
###
# use pct features (uncomment code)
# add interaction term to 4
#############################################
#################Feature Imp#################
library(xgboost)
xgb_dat= xgb.DMatrix(data=as.matrix(train),label = label)
############Select Features:
param1 <- list(objective = "reg:linear",booster='gbtree',
               eval_metric = "rmse",eta = 0.3)
imp_md = xgboost(param=param1,data = xgb_dat,nrounds = 100)

dat_name = colnames(train)



imp = xgb.importance(feature_names=dat_name,model=imp_md )
imp = as.data.frame(imp)
saveRDS(imp,'imp.RDS')
goodcols = as.character(imp$Feature)[1:45]

tr = train[,colnames(train) %in% goodcols]
tst = test[,colnames(test) %in% goodcols]

xgb_dat= xgb.DMatrix(data=as.matrix(tr),label = label)
###################XGBOOST GBTREE###########################

param1 <- list(objective = "reg:linear",booster='gbtree',
               eval_metric = "rmse",
               eta = 0.3
               # colsample_bytree=0.8,
               # subsample=0.8
               )
set.seed(1028)
bst.cv = xgb.cv(param=param1,data = xgb_dat, nfold = 10, nrounds = 3000,early.stop.round = 5)

#211 or
xgb_md = xgboost(param=param1,data = xgb_dat,nthread=4,nrounds = 559)
xgb_res1 = predict(xgb_md,as.matrix(tst))

sub2 = cbind(subm1,xgb_res1)
names(sub2)[3]='Purchase'
write.csv(sub2,file = 'sub2.csv')









# set.seed(1028)
# md0=lm(label~.,data=tr)
# res = predict(md0,tst)
sub2 = cbind(subm1,res)
names(sub2)[3]='Purchase'
write.csv(sub2,file = 'sub0.csv')
