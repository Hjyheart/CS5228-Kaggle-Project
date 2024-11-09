import xgboost
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from script import carly

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

####################### model training  ####################################
############################################################################
# Use the best_params from main-bayesian
best_params ={
    'alpha': 0.07272769267361982,
    'colsample_bytree': 0.5004695925119211,
    'early_stopping_rounds': 18,
    'gamma': 0.33307638017245794,
    'learning_rate': 0.014079290196807495,
    'max_depth': 52,
    'min_child_weight': 10,
    'n_estimators': 1123,
    'subsample': 0.575409364755193
}


xbg_model = xgboost.XGBRegressor(random_state           = 888888,
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
                                 verbosity = 2
                                )

xbg_model.fit(train_df_base[feature_list], train_df_base[target], eval_set=[(train_df_eval[feature_list], train_df_eval[target],)], verbose = False)

eval_rmse = mean_squared_error(train_df_eval[target], xbg_model.predict(train_df_eval[feature_list]), squared=False)
print(f"FINAL EVAL RMSE SCORE: {eval_rmse}")

submission_df = pd.DataFrame(xbg_model.predict(test_df[feature_list])).reset_index().rename(columns = {'index':"Id", 0:'Predicted'})
submission_df['Predicted'] = round(submission_df['Predicted'])
submission_df.to_csv(fr"./dataset/{model_name}_submission.csv", index = False)
