
# coding: utf-8

# In[3]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

train_df = pd.read_csv('train_bnp.csv')
test_df = pd.read_csv('test_bnp.csv')

train_size = train_df.shape[0]

overall_data = train_df.append(test_df, ignore_index = True)

num_fields = overall_data._get_numeric_data()
for j in list(num_fields):
    overall_data[j] = overall_data[j].fillna(-1)
    


# In[7]:

train_df.head(10)


# In[3]:

# ###################################TESTING AND LOOKING AT DATA
# import seaborn as sns
# num_fields = train_df._get_numeric_data()
# # all_fields = train_df.columns.values

# ##NUMERICAL VARIABLE ANALYSIS
# for j in list(num_fields):
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.hist(train_df[j], bins = 10, range = (train_df[j].min(),train_df[j].max()))
#     plt.title(j)
#     plt.xlabel(j)
#     plt.ylabel('Count')
#     plt.show()
    
#     train_df.boxplot(column = j)
    
#     sns.boxplot(y = j, data = train_x)
print "a"
    


# In[4]:

import numpy as np
from sklearn.base import TransformerMixin

####DATA IMPUTER - THIS WORKS ON BOTH NUMERIC AND CATEGORICAL FIELDS. THIS USES MEDIAN FOR NUMERIC AND MOST FREQUENT FOR CATEGORICAL
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


overall_data = DataFrameImputer().fit_transform(overall_data)




# In[5]:

train_df = overall_data.ix[:train_size-1,:]
train_df = train_df.drop('ID', axis = 1)

test_df = overall_data.ix[train_size:,:]
test_df = test_df.drop('target',axis = 1)
test_df = test_df.drop('ID', axis = 1)


# In[6]:

irregular_fields = ['v22','v47','v56','v71','v79','v113','v125']

for k in list(irregular_fields):
    print k
    from copy import deepcopy
    #list of variable
    var_list =sorted(train_df[k].unique())

    #list of target
    targets=sorted(train_df["target"].unique())

    #count of targets
    target_count=train_df.groupby(["target"]).size()

    #count of variable
    var_counts=train_df.groupby([k]).size()

    #count of variable cross target
    var_target_counts=train_df.groupby([k,"target"]).size()

    ### calculating log odds for train dataset
    logodds={}
    logoddsPA={}
    MIN_CAT_COUNTS=2
    default_logodds=np.log(target_count/len(train_df))-np.log(1.0-target_count/float(len(train_df)))
    for addr in var_list:
        PA=var_counts[addr]/float(len(train_df))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        for cat in var_target_counts[addr].keys():
            if (var_target_counts[addr][cat]>MIN_CAT_COUNTS) and var_target_counts[addr][cat]<var_counts[addr]:
                PA=var_target_counts[addr][cat]/float(var_counts[addr])
                logodds[addr][targets.index(cat)]=np.log(PA)-np.log(1.0-PA)
        logodds[addr]=pd.Series(logodds[addr])
        logodds[addr].index=range(len(targets))

    #merge logodds with train data
    train_df["logoddsPA"]=train_df[k].apply(lambda x: logoddsPA[x])

    c123=train_df[k].apply(lambda x: logodds[x])
    c123.columns=["logodds_"+str(k)+str(x) for x in range(len(c123.columns))]

    train_df = train_df.join(c123.ix[:,:])
    train_df = train_df.drop(k, axis = 1)


# In[50]:

for k in list(irregular_fields):
    print k
    #logodds for test dataset
    new_var_list = sorted(test_df[k].unique())
    new_var_counts = test_df.groupby(k).size()

    only_new=set(new_var_list+var_list)-set(var_list)
    only_old=set(new_var_list+var_list)-set(new_var_list)
    in_both=set(new_var_list).intersection(var_list)

    for addr in only_new:
        PA=new_var_counts[addr]/float(len(test_df)+len(train_df))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        logodds[addr].index=range(len(targets))
    for addr in in_both:
        PA=(var_counts[addr]+new_var_counts[addr])/float(len(test_df)+len(train_df))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)    

    # merge logodds with test data
    test_df["logoddsPA"]=test_df[k].apply(lambda x: logoddsPA[x])

    c123=test_df[k].apply(lambda x: logodds[x])
    c123.columns=["logodds_"+str(k)+str(x) for x in range(len(c123.columns))]

    test_df = test_df.join(c123.ix[:,:])
    test_df = test_df.drop(k, axis = 1)


# In[4]:

num_fields = train_df._get_numeric_data()
all_fields = train_df.columns.values
cat_fields = list(set(all_fields) - set(num_fields))

train_df = pd.get_dummies(train_df, columns = cat_fields)

num_fields = test_df._get_numeric_data()
all_fields = test_df.columns.values
cat_fields = list(set(all_fields) - set(num_fields))
test_df = pd.get_dummies(test_df, columns = cat_fields)


# In[52]:

train_x = train_df.ix[:, train_df.columns != 'target']
train_y = train_df.ix[:, 'target']



# In[53]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_df = scaler.transform(test_df)


# In[81]:



from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import xgboost as xgb

sss = StratifiedShuffleSplit(train_y, train_size=0.5)
for train_index, test_index in sss:
    train_x_train,train_x_test=train_x[train_index],train_x[test_index]
    train_y_train,train_y_test=train_y[train_index],train_y[test_index]
    

# xg_train = xgb.DMatrix(train_x_train, label = train_y_train)
# xg_test = xgb.DMatrix(train_x_test, label = train_y_test)

# param = {}
# # use softmax multi-class classification
# param['objective'] = 'reg:logistic'
# # scale weight of positive examples
# param['eta'] = 0.1
# param['max_depth'] = 6
# param['min_child_weight'] = 3,
# param['max_delta_step'] = 5,
# param['subsample'] = 0.5,
# # param['colsample_bytree'] = 0.7,
# param['n_estimators'] = 300,
# param['learning_rate'] = 0.05
# param['silent'] = 1
# param['nthread'] = 4

# watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
# num_round = 5
# bst = xgb.train(param, xg_train, num_round, watchlist )
# # get prediction
# pred = bst.predict( xg_test )

# model = RandomForestClassifier(n_jobs = -1,max_features = "auto", n_estimators = 400,max_depth = 40, min_samples_leaf = 200
#                                ,min_samples_split = 40, criterion = 'gini',class_weight = 'auto' )
# model = LogisticRegression()
model = XGBClassifier(objective= "reg:logistic",
                      max_depth=6, 
                      learning_rate = 0.025,
                      n_estimators=300, 
                      silent = True,
                      nthread = -1,
                      gamma = 0,
                      min_child_weight = 3,
                      max_delta_step = 5,
                      subsample = 0.8,
                      colsample_bytree = 0.7,          
                      )

model.fit(train_x_train,train_y_train)

# c123 = model.predict(features_test)

# # In[24]:

print("all", log_loss(train_y, model.predict_proba(train_x)))
print("train", log_loss(train_y_train, model.predict_proba(train_x_train)))
print("test", log_loss(train_y_test, model.predict_proba(train_x_test)))


# In[82]:

preds = model.predict_proba(test_df)
preds_df = pd.DataFrame(preds)
preds_df.to_csv('preds_bnp.csv')


# In[71]:

pred


# In[ ]:

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier

xgb = XGBClassifier()
xgb_params = {
    'objective':["reg:logistic"],
    'n_estimators': [100,200,300,400],
    'max_depth': [3,6,9],
    'learning_rate': [0.01,0.05],
    'nthread' :[-1],
    'min_child_weight': [1,2,3,4],
    'max_delta_step': [3,4,5,6],
    'subsample':[0.5,0.6,0.7,0.8],
    'colsample_bytree' : [0.6,0.7,0.8]
}
model = GridSearchCV(xgb, xgb_params, n_jobs=-1, cv=2,scoring='log_loss',verbose = 20)

# rf = RandomForestClassifier()
# rf_params = {
#     'n_estimators': [100,200,300,400],
#     'max_depth': [10,20,30,40],
#     'class_weight' : ["auto"],
#     'min_samples_leaf':[100,200]
# }
# model = GridSearchCV(rf, rf_params, n_jobs=-1, cv=2,scoring='log_loss',verbose = 20)


model.fit(train_x,train_y)
# cv = cross_val_score(model , train_x, train_y)


print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)


# In[15]:



