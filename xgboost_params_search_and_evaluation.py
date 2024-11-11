import pickle
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce

from skopt import BayesSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  root_mean_squared_error, mean_absolute_error, r2_score

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
    input_df    = carly.dataproc.FIXNULL_fueltype(df = input_df)
    
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

train_df_RAW    = pd.read_csv("./dataset/train.csv")
test_df_RAW     = pd.read_csv("./dataset/test.csv")
train_df        = model_preprocessing(input_df = train_df_RAW)
test_df         = model_preprocessing(input_df = test_df_RAW)

# ### Remove outliers from training dataset
train_df        = train_df.loc[~(train_df['AGE-remaining']<0) | ~(train_df['AGE-currentREG']<0)]
outlier_df      = carly.outlier_detection.identify_outliers(df=train_df)
train_df        = outlier_df.loc[(outlier_df['cluster']!=-1) & (outlier_df['CAT-rare&exotic']!=1)].drop(columns=['cluster'])

# ### Bayesian Search
name = "bayesian-search"
train_df_base, train_df_eval = train_test_split(train_df, test_size=0.2, random_state=888888)

####################### hyperparameters tuning ######################
#####################################################################

# Define the hyperparameter distributions
param_dist = {
            'n_estimators'          : (300,  2000),
            'max_depth'             : (3, 200),
            'learning_rate'         : (0.01, 0.1, "log-uniform"),
            'gamma'                 : (0.01, 0.9),    
            'alpha'                 : (0.01, 0.1),     
            
            'subsample'             : (0.3, 0.7),
            'colsample_bytree'      : (0.3, 0.7),
            'early_stopping_rounds' : (8, 18), 
            'min_child_weight'      : (3, 88),   
            }

xgb_model       = xgboost.XGBRegressor(random_state = 888888, n_jobs = -1)

# Create the RandomizedSearchCV object
bayesian_search = BayesSearchCV(xgb_model,
                                search_spaces  = param_dist,
                                n_iter               = 1,
                                cv                   = 5,
                                random_state         = 888888,
                                scoring              = 'neg_root_mean_squared_error',)

# Fit the RandomizedSearchCV object to the training data
bayesian_search.fit(train_df_base[carly.params.ENCODED_feature_list], 
                    train_df_base[carly.params.target], 
                    eval_set=[(train_df_eval[carly.params.ENCODED_feature_list], train_df_eval[carly.params.target],)], 
                    verbose = False)

ENCODED_bayesian_search_df = pd.DataFrame(bayesian_search.cv_results_)
ENCODED_bayesian_search_df = ENCODED_bayesian_search_df[['params',
                                                         'split0_test_score',
                                                         'split1_test_score',
                                                         'split2_test_score',
                                                         'split3_test_score',
                                                         'split4_test_score',
                                                         'mean_test_score',
                                                         'std_test_score',
                                                         'rank_test_score']]

ENCODED_bayesian_search_df['objfunc'] = ENCODED_bayesian_search_df[['split0_test_score',
                                                                    'split1_test_score',
                                                                    'split2_test_score',
                                                                    'split3_test_score',
                                                                    'split4_test_score',]].median(axis=1)
ENCODED_bayesian_search_df = ENCODED_bayesian_search_df.sort_values('objfunc', ascending = False)

best_params = ENCODED_bayesian_search_df['params'].iloc[0]
print(best_params)

#############################################################
####################### this is the set of best params, used in main.py
#############################################################

# best_params ={'alpha': 0.07272769267361982,
#              'colsample_bytree': 0.5004695925119211,
#              'early_stopping_rounds': 18,
#              'gamma': 0.33307638017245794,
#              'learning_rate': 0.014079290196807495,
#              'max_depth': 52,
#              'min_child_weight': 10,
#              'n_estimators': 1123,
#              'subsample': 0.575409364755193}

#############################################################
########## Further analysis of xgboost model 
#############################################################

ENCODED_xbg_model = xgboost.XGBRegressor(random_state           = 888888,
                                         n_estimators           = best_params['n_estimators'],
                                         n_jobs                 = -1,
                                         max_depth              = best_params['max_depth'],
                                         early_stopping_rounds  = best_params['early_stopping_rounds'],
                                         
                                         learning_rate          = best_params['learning_rate'], #  step size shrinkage used in updates to prevent overfitting. Lower values make the model more robust by taking smaller steps.
                                         gamma                  = best_params['gamma'], #minimum loss reduction required to make a further partition on a leaf node of the tree. Higher values increase the regularization.
                                         alpha                  = best_params['alpha'], #L2 regularization term on weights. 
                                         
                                         subsample              = best_params['subsample'], #percentage of rows used for each tree construction. 
                                         colsample_bytree       = best_params['colsample_bytree'], #percentage of columns used for each tree construction
                                         min_child_weight       = best_params['min_child_weight'], #Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                                         verbosity              = 2)

ENCODED_xbg_model.fit(train_df_base[carly.params.ENCODED_feature_list], train_df_base[carly.params.target],
                      eval_set=[(train_df_eval[carly.params.ENCODED_feature_list], train_df_eval[carly.params.target],)], verbose = False)

eval_rmse = root_mean_squared_error(train_df_eval[carly.params.target], ENCODED_xbg_model.predict(train_df_eval[carly.params.ENCODED_feature_list]))
print(f"FINAL EVAL RMSE SCORE: {eval_rmse}")

eval_mae = mean_absolute_error(train_df_eval[carly.params.target], ENCODED_xbg_model.predict(train_df_eval[carly.params.ENCODED_feature_list]))
print(f"FINAL EVAL MAE SCORE: {eval_mae}")

eval_r2score = r2_score(train_df_eval[carly.params.target], ENCODED_xbg_model.predict(train_df_eval[carly.params.ENCODED_feature_list]))
print(f"FINAL EVAL R2 SCORE: {eval_r2score}")


submission_df               = pd.DataFrame(ENCODED_xbg_model.predict(test_df[ENCODED_feature_list])).reset_index().rename(columns = {'index':"Id", 0:'Predicted'})
submission_df['Predicted']  = round(submission_df['Predicted'])

feature_importance = ENCODED_xbg_model.feature_importances_

importance_df = pd.DataFrame({'Feature'     : ENCODED_feature_list,
                              'Importance'  : feature_importance})

# step 1: Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
xgboost.plot_importance(ENCODED_xbg_model, importance_type='weight', max_num_features=20)  # You can also use 'gain' or 'cover'
plt.title("Feature Importance from XGBoost")
plt.show()

# step 2: plot scatter plot 
y_pred = ENCODED_xbg_model.predict(train_df_eval[ENCODED_feature_list])
residuals = train_df_eval[target] - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# step 3: plot histogram of errors
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', color='skyblue')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.show()


# step 3: plot histogram of errors
low_price   = train_df_eval[target][train_df_eval[target] < 15000]
mid_price   = train_df_eval[target][(train_df_eval[target] >= 15000) & (train_df_eval[target] < 30000)]
high_price  = train_df_eval[target][train_df_eval[target] >= 30000]

low_pred    = y_pred[train_df_eval[target] < 15000]
mid_pred    = y_pred[(train_df_eval[target] >= 15000) & (train_df_eval[target] < 30000)]
high_pred   = y_pred[train_df_eval[target] >= 30000]

# step 4.1: Calculate MAE and RMSE for each segment
segments    = ['Low Price', 'Mid Price', 'High Price']
mae_values  = [mean_absolute_error(low_price, low_pred),
               mean_absolute_error(mid_price, mid_pred),
               mean_absolute_error(high_price, high_pred)
              ]
rmse_values = [root_mean_squared_error(low_price, low_pred),
               root_mean_squared_error(mid_price, mid_pred),
               root_mean_squared_error(high_price, high_pred)
               ]

x = np.arange(len(segments))

# step 4.2: plot bar graph for across segments
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, mae_values, width=0.4, label='MAE', color='lightcoral')
plt.bar(x + 0.2, rmse_values, width=0.4, label='RMSE', color='skyblue')
plt.xticks(x, segments)
plt.xlabel("Price Segment")
plt.ylabel("Error")
plt.title("MAE and RMSE by Price Segment")
plt.legend()
plt.show()
