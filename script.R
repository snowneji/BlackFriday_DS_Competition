library(data.table)
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
train = fread('train.csv',data.table = F)
test = fread('test-comb.csv',data.table = F)
test = test %>% select(-V1,-Comb)
subm1 =  test %>% select(User_ID,Product_ID)
#Occupation
train$Occupation = as.numeric(as.factor(train$Occupation))
test$Occupation = as.numeric(as.factor(test$Occupation))
#Marital Status
train$Marital_Status = as.numeric(as.factor(train$Marital_Status))
test$Marital_Status = as.numeric(as.factor(test$Marital_Status))
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
Purchase = train$Purchase
train = train %>% select(-Purchase)
all = rbind(train,test)
all$User_ID = as.numeric(as.factor(all$User_ID))
all$Product_ID = as.numeric(as.factor(all$Product_ID))

train = all[1:nrow(train),]
train = cbind(Purchase,train)
test = all[(nrow(train)+1):nrow(all),]
rm(Purchase,all)




# mean purchase for each person
person_mean = train %>% group_by(User_ID) %>% summarise(p_pmean=round(mean(Purchase)))
train = left_join(train,person_mean, by=c('User_ID'='User_ID'))
test = left_join(test,person_mean, by=c('User_ID'='User_ID'))
rm(person_mean)
# mean purchase for each product
pro_pmean = train %>% group_by(Product_ID) %>% summarise(pro_pmean=round(mean(Purchase)))
train = left_join(train,pro_pmean, by=c('Product_ID'='Product_ID'))
test = left_join(test,pro_pmean, by=c('Product_ID'='Product_ID'))
test[is.na(test$pro_pmean),]$pro_pmean = 0
rm(pro_pmean)

# mean purchase for each Age
age_pmean = train %>% group_by(Age) %>% summarise(age_pmean=round(mean(Purchase)))
train = left_join(train,age_pmean,by=c('Age'='Age'))
test = left_join(test,age_pmean,by=c('Age'='Age'))
rm(age_pmean)
# mean purchase for each Occupation
occu_pmean = train %>% group_by(Occupation) %>% summarise(occu_pmean=round(mean(Purchase)))
train = left_join(train,occu_pmean,by=c('Occupation'='Occupation'))
test = left_join(test,occu_pmean,by=c('Occupation'='Occupation'))
rm(occu_pmean)
#
#mean purchase for each Gender
gender_pmean = train %>% group_by(Gender) %>% summarise(gender_pmean=round(mean(Purchase)))
train = left_join(train,gender_pmean,by=c('Gender'='Gender'))
test = left_join(test,gender_pmean,by=c('Gender'='Gender'))
rm(gender_pmean)
#mean purchase for each city
city_pmean = train %>% group_by(City_Category) %>% summarise(city_pmean=round(mean(Purchase)))
train = left_join(train,city_pmean,by=c('City_Category'='City_Category'))
test = left_join(test,city_pmean,by=c('City_Category'='City_Category'))
rm(city_pmean)
#mean purchase for each stay_year
stay_pmean = train %>% group_by(Stay_In_Current_City_Years) %>% summarise(stay_pmean=round(mean(Purchase)))
train = left_join(train,stay_pmean,by=c('Stay_In_Current_City_Years'='Stay_In_Current_City_Years'))
test = left_join(test,stay_pmean,by=c('Stay_In_Current_City_Years'='Stay_In_Current_City_Years'))
rm(stay_pmean)






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


train$catm12 = train$Product_Category_1 * train$Product_Category_2
train$catm13 = train$Product_Category_1 * train$Product_Category_3
train$catm23 = train$Product_Category_2 * train$Product_Category_3
train$catm123 = train$Product_Category_1 * train$Product_Category_2 * train$Product_Category_3

test$catm12 = test$Product_Category_1 * test$Product_Category_2
test$catm13 = test$Product_Category_1 * test$Product_Category_3
test$catm23 = test$Product_Category_2 * test$Product_Category_3
test$catm123 = test$Product_Category_1 * test$Product_Category_2 * test$Product_Category_3


#############aggregate features:
######################
label = train$Purchase
# train = train %>% select(-label)
train = train %>% select(-Purchase)
######################
all=rbind(train,test)
#each user product:
each_user_prd = all %>% group_by(User_ID) %>% summarise(each_user_prd = length(unique(Product_ID)))
train = left_join(train,each_user_prd,by=c('User_ID' = 'User_ID'))
test = left_join(test,each_user_prd,by=c('User_ID' = 'User_ID'))
rm(each_user_prd)
#each product n-user:
each_prd_user = all %>% group_by(Product_ID) %>% summarise(each_prd_user = length(unique(User_ID)))
train = left_join(train,each_prd_user,by=c('Product_ID' = 'Product_ID'))
test = left_join(test,each_prd_user,by=c('Product_ID' = 'Product_ID'))
rm(each_prd_user)




xgb_dat= xgb.DMatrix(data=as.matrix(train),label = label)
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
#17:        cv: 2505      lb:2512       (using top 40 features)
#18:        cv: 2506      lb:2507       (using top 45 features)
############## add mean purchases features
#19:        cv: 2447.22   lb:2495.68     (using top 45 features)

##############no interaction and other features
#20         cv: 2455    lb:2493     (using top 5 features)
##############aggregate features
#20         cv: 2440.9    lb:2477           (using top 9 features)
#21         cv: 2438      lb:2472           (using top 24 features)


###
#Next Step
###
# use pct features (uncomment code)
# add interaction term to 4
#############################################
#################Feature Imp#################
############Select Features:
param1 <- list(objective = "reg:linear",booster='gbtree',
               eval_metric = "rmse",eta = 0.3,
               colsample_bytree=0.8,
               subsample=0.8)
imp_md = xgboost(param=param1,data = xgb_dat,nrounds = 100)

dat_name = colnames(train)



imp = xgb.importance(feature_names=dat_name,model=imp_md )
imp = as.data.frame(imp)
saveRDS(imp,'imp.RDS')
goodcols = as.character(imp$Feature)

tr = train[,colnames(train) %in% goodcols]
tst = test[,colnames(test) %in% goodcols]

xgb_dat= xgb.DMatrix(data=as.matrix(tr),label = label)
###################XGBOOST GBTREE###########################

param1 <- list(objective = "reg:linear",booster='gbtree',
               eval_metric = "rmse",
               eta = 0.4
               # colsample_bytree=0.8,
               # subsample=0.8
               )
set.seed(1028)
bst.cv = xgb.cv(param=param1,data = xgb_dat, nfold = 10, nrounds = 3000,early.stop.round = 5)

#211 or
xgb_md = xgboost(param=param1,data = xgb_dat,nthread=4,nrounds = 541)
xgb_res1 = predict(xgb_md,as.matrix(tst))

sub2 = data.frame(subm1, Purchase= as.numeric(xgb_res1))
sub2 = as.data.frame(sub2)
names(sub2)[3]='Purchase'
sub2$Purchase = as.numeric(sub2$Purchase)

write.csv(sub2,file = 'sub5.csv',row.names = F)








# set.seed(1028)
# md0=lm(label~.,data=tr)
# res = predict(md0,tst)
sub2 = cbind(subm1,res)
names(sub2)[3]='Purchase'
write.csv(sub2,file = 'sub0.csv')
