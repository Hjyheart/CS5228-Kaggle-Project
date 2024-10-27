import sys
import os
import difflib
import datetime
import dateutil
import xgboost
import numpy as np
import pandas as pd
import scipy.stats as stats
import category_encoders as ce

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, os.getcwd() + '/script/')
import carly

def model_preprocessing(input_df):
    
    input_df = input_df.drop(columns = ['eco_category', # same value for all records
                                        'indicative_price', # same value for all records
                                        'lifespan', #more than 90% of records null values 
                                       ])
    
    ####################### data processing ######################
    ##############################################################
    
    input_df    = carly.dataproc.DEDUP_listingid(df = input_df)
    
    input_df    = carly.dataproc.FIXNULL_make(df = input_df)
    input_df    = carly.dataproc.FIXNULL_manufacture(df = input_df)
    input_df    = carly.dataproc.FIXNULL_ownercount(df = input_df)
    input_df    = carly.dataproc.FIXNULL_dereg(df = input_df)
    
    input_df    = carly.dataproc.ENCODE_category(df = input_df, drop = False)
    input_df    = carly.dataproc.ENCODE_transmission(df = input_df, drop=True)
    input_df    = carly.dataproc.ENCODE_vehtype(df = input_df, drop=False)
    input_df    = carly.dataproc.ENCODE_make(df = input_df, drop=False)
    input_df    = carly.dataproc.ENCODE_fueltype(df = input_df, drop=False)
    input_df    = carly.dataproc.ENCODE_model(df = input_df, drop=False)
    input_df    = carly.dataproc.ENCODE_opcscheme(df = input_df)
    
    input_df    = carly.dataproc.EXTRACT_features(df = input_df)
    input_df    = carly.dataproc.EXTRACT_desc(df = input_df)
    input_df    = carly.dataproc.EXTRACT_accessories(df = input_df)
    
    input_df    = carly.dataproc.CALC_currentage(df = input_df)
    input_df    = carly.dataproc.CALC_remainage(df = input_df)
    
    input_df    = carly.dataproc.IMPUTENULL_power(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_curbweight(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_enginecap(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_depreciation(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_omv(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_arf(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_mileage(df = input_df)
    input_df    = carly.dataproc.IMPUTENULL_roadtax(df = input_df)
    
    input_df    = carly.dataproc.RATIO_ownertoage(df = input_df)
    input_df    = carly.dataproc.RATIO_miletoage(df = input_df)
    input_df    = carly.dataproc.RATIO_curbtopower(df = input_df)
    input_df    = carly.dataproc.RATIO_indirectcost(df = input_df)
    input_df    = carly.dataproc.RATIO_ARFtoOMV(df = input_df)
    input_df    = carly.dataproc.RATIO_ARFtoDEREG(df = input_df)
    input_df    = carly.dataproc.RATIO_miletoowner(df = input_df)

    ### enable categorical without encoding ######################
    ##############################################################
    
    cat_features = ["make", "type_of_vehicle", "fuel_type", "model", "CAT-elechybrid", "CAT-coeparf",]
    input_df[cat_features] = input_df[cat_features].astype('category')
    
    return input_df

####################### data processing ######################
##############################################################

train_df_RAW                    = pd.read_csv(r"./data/train.csv")
test_df_RAW                     = pd.read_csv(r"./data/test.csv")
cat_encoding                    = True

train_df                        = model_preprocessing(input_df = train_df_RAW)
train_df_base, train_df_eval    = train_test_split(train_df, test_size=0.2, random_state=888888)
test_df                         = model_preprocessing(input_df = test_df_RAW)

if cat_encoding == False:
    feature_list        = carly.params.CAT_feature_list
    target              = carly.params.target
    enable_categorical  = True
    model_name          = 'CAT_xgboost'
    
elif cat_encoding == True: 
    feature_list        = carly.params.ENCODED_feature_list 
    target              = carly.params.target    
    enable_categorical  = False
    model_name          = 'ENCODED_xgboost'

####################### hyperparameters tuning ######################
#####################################################################

# Define the hyperparameter distributions
param_dist = {
            'learning_rate'         : stats.uniform(0.01, 0.1),
            'gamma'                 : stats.uniform(0.01, 0.9),    
            'alpha'                 : stats.uniform(0.01, 0.1),
            'subsample'             : stats.uniform(0.3, 0.7),
            'colsample_bytree'      : stats.uniform(0.3, 0.7),
            'early_stopping_rounds' : stats.randint(8, 18), 
            'min_child_weight'      : stats.randint(3, 88),   
            }

xgb_model = xgboost.XGBRegressor(random_state       = 888888, 
                                 n_estimators       = 1888,
                                 max_depth          = 5,              
                                 enable_categorical = enable_categorical,
                                 n_jobs             = -1,
                                 )

random_search = RandomizedSearchCV(xgb_model, 
                                   param_distributions  = param_dist, 
                                   n_iter               = 88, 
                                   cv                   = 5, 
                                   random_state         = 888888,
                                   scoring              = 'neg_root_mean_squared_error',)

random_search.fit(train_df_base[feature_list], train_df_base[target], eval_set = [(train_df_eval[feature_list], train_df_eval[target],)], verbose = False)
random_search_df = pd.DataFrame(random_search.cv_results_)
random_search_df = random_search_df[['params',
                                    'split0_test_score',
                                    'split1_test_score',
                                    'split2_test_score',
                                    'split3_test_score',
                                    'split4_test_score',
                                    'mean_test_score',
                                    'std_test_score',
                                    'rank_test_score']]

random_search_df['objfunc'] = random_search_df[['split0_test_score',
                                                'split1_test_score',
                                                'split2_test_score',
                                                'split3_test_score',
                                                'split4_test_score',]].median(axis=1)
random_search_df = random_search_df.sort_values('objfunc', ascending = False)
best_params = random_search_df['params'].iloc[0]

####################### model training  ####################################
############################################################################

xbg_model = xgboost.XGBRegressor(random_state          = 888888,
                                n_estimators           = 1888,
                                enable_categorical     = True,
                                n_jobs                 = -1,
                                max_depth              = 5,
                                early_stopping_rounds  = best_params['early_stopping_rounds'],
                                
                                learning_rate          = best_params['learning_rate'], #  step size shrinkage used in updates to prevent overfitting. Lower values make the model more robust by taking smaller steps.
                                gamma                  = best_params['gamma'], #minimum loss reduction required to make a further partition on a leaf node of the tree. Higher values increase the regularization.
                                alpha                  = best_params['alpha'], #L2 regularization term on weights. 
                                
                                subsample              = best_params['subsample'], #percentage of rows used for each tree construction. 
                                colsample_bytree       = best_params['colsample_bytree'], #percentage of columns used for each tree construction
                                min_child_weight       = best_params['min_child_weight'], #Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                                verbosity = 2
                                )

xbg_model.fit(train_df_base[feature_list], train_df_base[target], eval_set=[(train_df_eval[feature_list], train_df_eval[target],)], verbose = False)

eval_rmse = mean_squared_error(train_df_eval[target], xbg_model.predict(train_df_eval[feature_list]), squared=False)
print(f"FINAL EVAL RMSE SCORE: {eval_rmse}")

submission_df = pd.DataFrame(xbg_model.predict(test_df[feature_list])).reset_index().rename(columns = {'index':"Id", 0:'Predicted'})
submission_df['Predicted'] = round(submission_df['Predicted'])
submission_df.to_csv(fr"./data/{model_name}_submission.csv", index = False)
