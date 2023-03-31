import numpy as np
#from forOwn import RCCA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import sys
import random
import matplotlib.pyplot as plt
from cca_zoo.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from customrCCA import ReferenceCCA_IPLS



df_pca_gray=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/PCA__components_gray.csv').iloc[:,1:]

df_pca_white=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/PCA__components_white.csv').iloc[:,1:]

df_cognition=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/Cognition_Uncorrected_Subtract_Result_data_all_replace_nan_zero_normalized.csv')['nihtbx_picvocab_uncorrected']

df_cognition = np.nan_to_num(df_cognition, nan=0)
X_train, X_test, y_train, y_test, ref_train, ref_test = train_test_split(df_pca_gray, df_pca_white, df_cognition, test_size=0.3, random_state=50)

ref_train = np.array(ref_train).reshape(-1,1)
ref_test = np.array(ref_test).reshape(-1,1)


def scorer(estimator,X):
    dim_corrs=estimator.score(X) #Goes inside _basemodel.py score function
    return dim_corrs.mean()

#c1 = [0.0007,0.0008,0.0009,0.001]
#c2 = [0.0007,0.0008,0.0009,0.001]
c1 = [0.0005]
c2 = [0.0005]
c3 = [0.04]#Optional

lr = 0.00001
cv = 5 #number of folds


#-----USAGE without GridSearch - UNCOMMENT TO USE---------------------------------------------------

# model=SCCA_IPLS(c=[0.01,0.01,0.001], latent_dims=3,verbose=True, lr=0.001)
# referenceCCAModel=model.fit([X_train,y_train, ref_train])
# train_scores = referenceCCAModel.score([X_train, y_train, ref_train])
# test_scores = referenceCCAModel.score([X_test, y_test, ref_test])

#------END-----------------------------------------------------------------------


#-----USAGE with GridSearch---------------------------------------------------

# model=GridSearchCV(SCCA_IPLS(latent_dims=3, verbose=True, lr=lr),param_grid=param_grid,scoring=scorer)
# referenceCCAModel = model.fit([X_train,y_train])
# train_scores = referenceCCAModel.best_estimator_.score([X_train,y_train]) ##Goes inside _basemodel.py score function
# test_scores=referenceCCAModel.best_estimator_.score([X_test, y_test]) ##Goes inside _basemodel.py score function

#------END-----------------------------------------------------------------------

#-----USAGE with GridSearch Reference---------------------------------------------------

param_grid = {'c': [c1,c2,c3]}
model=GridSearchCV(ReferenceCCA_IPLS(latent_dims=3, verbose=True, lr=lr),param_grid=param_grid,scoring=scorer)
referenceCCAModel = model.fit([X_train,y_train,ref_train])
train_scores= referenceCCAModel.best_estimator_.score([X_train,y_train,ref_train]) 
test_scores = referenceCCAModel.best_estimator_.score([X_test, y_test,ref_test]) 

#------END-----------------------------------------------------------------------
##df=pd.DataFrame(referenceCCAModel.cv_results_)
##df=df.sort_values("rank_test_score")
#print(df)
##df.to_csv("Grid_Best_Estimator.txt", sep="\t")

print(referenceCCAModel.best_estimator_)

print(train_scores)
#print(cor_1)
print(test_scores)
#print(cor_2t)

#Transform training and finding correlation and p value
# t1,t2=model.best_estimator_.transform([X_train,y_train])

# from scipy import stats
# train_accept=[]
# for i in range (0, 80):
#     r_train,p_train=stats.pearsonr(t1[:,i], t2[:,i])
#     print("correlation",round(r_train,4),"P-Value",p_train)
#     if (p_train<=0.000625):
#         train_accept.append(i)

# print(train_accept)

#Transform testing and finding correlation and p value
# t3,t4=model.best_estimator_.transform([X_test,y_test])

# from scipy import stats
# test_accept=[]
# for i in range (0, 80):
#     r_test,p_test=stats.pearsonr(t3[:,i], t4[:,i])
#     print("correlation",round(r_test,4),"P-Value",p_test)
#     if (p_test<=0.000625):
#         test_accept.append(i)

# print(test_accept)

#Correlation and p value between gray matter and reference
# for i in range(0,611):
#     corr, p_value = pearsonr(X_test[:, i], ref_test)
#     print(f"Pearson correlation: {corr:.3f}")
#     print(f"P-value: {p_value:.3f}")

# for i in range(0,611):
#     corr, p_value = pearsonr(y_test[:, i], ref_test)
#     print(f"Pearson correlation: {corr:.3f}")
#     print(f"P-value: {p_value:.3f}")

