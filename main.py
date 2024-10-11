import pandas as pd
import difflib
import datetime
import dateutil
import xgboost
import category_encoders as ce
import scipy.stats as stats
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV

def DEDUP_listingid(df):
    
    df['listing_id']    = df['listing_id'].astype(int).astype(str).str.strip()
    df                  = df.drop_duplicates('listing_id')

    return (df)
    
def FIXNULL_make(df): 

    df['make']          = df['make'].str.upper()
    df['make']          = df['make'].str.replace(' ','').str.strip()
    df['title']         = df['title'].str.upper()
    
    make_list           = [make for make in df['make'].unique().tolist() if type(make) == str]
    df['make_temp']     = df['title'].str.split(' ').str[0]
    df['make_temp']     = df.apply(lambda x: difflib.get_close_matches(x['make_temp'], make_list, n=1)[0], axis=1)
    df['make']          = df['make'].fillna(df['make_temp'])
    df                  = df.drop(columns = 'make_temp')

    return (df)

def FIXNULL_manufacture(df):

    df['original_reg_date']         = pd.to_datetime(df['original_reg_date'], format = "%d-%b-%Y")
    df['reg_date']                  = pd.to_datetime(df['reg_date'], format = "%d-%b-%Y")
    
    df['original_reg_date_temp']    = df['original_reg_date'].dt.year
    df['reg_date_temp']             = df['reg_date'].dt.year
    df['manufactured']              = df['manufactured'].fillna(df[['original_reg_date_temp','reg_date_temp']].min(axis=1))
    df['manufactured']              = df['manufactured'].astype(int).astype(str)
    df                              = df.drop(columns = ['original_reg_date_temp', 'reg_date_temp'])
    
    return (df)

def FIXNULL_ownercount(df):
    
    df['no_of_owners'] = df['no_of_owners'].fillna(0)
    
    return df

def FIXNULL_dereg(df):
    df['dereg_value'] = df['dereg_value'].fillna(0)
    return(df)

def ENCODE_category(df):
    
    df_temp             = df['category'].str.split(',', expand = True)
    df_temp             = df_temp.astype(str)
    category_set        = set(sum(df['category'].str.split(',').tolist(), []))
    category_set        = set([category.strip() for category in category_set if len(category.strip()) > 1])
    category_dflist     = [df_temp.apply(lambda x: sum(x.str.contains(category)), axis=1).rename(f"CAT-{category.replace(' ','')}") for category in category_set]
    df                  = pd.concat([df]+category_dflist, axis = 1)
    df                  = df.drop(columns = ['category'])
    return (df)

def ENCODE_transmission(df):
    
    df['TRANSMISSION-manual'] = pd.get_dummies(df['transmission'], drop_first=True, dtype=int)
    df                        = df.drop(columns = ['transmission'])
    
    return (df)

def ENCODE_vehtype(df):
    
    encoder             = ce.BinaryEncoder(cols='type_of_vehicle',return_df=True)
    df_temp             = encoder.fit_transform(df['type_of_vehicle']) 
    df_temp.columns     = [col.replace('type_of_vehicle_', "TYPE-binenc") for col in df_temp.columns] 

    df                  = pd.concat([df, df_temp], axis = 1)
    df                  = df.drop(columns = 'type_of_vehicle')

    return df

def ENCODE_make(df):

    encoder             = ce.BinaryEncoder(cols='make',return_df=True)
    df_temp             = encoder.fit_transform(df['make']) 
    df_temp.columns     = [col.replace('make_', "MAKE-binenc") for col in df_temp.columns] 

    df = pd.concat([df, df_temp], axis =1)
    df = df.drop(columns = ['make'])
    
    return(df)

def ENCODE_opcscheme(df): 
    df.loc[~(df['opc_scheme'].isnull()),'opc_scheme'] = 1
    df['opc_scheme'] = df['opc_scheme'].fillna(0)

    return df

def ENCODE_fueltype(df):
    
    encoder         = ce.BinaryEncoder(cols='fuel_type',return_df=True)
    df_temp         = encoder.fit_transform(df['fuel_type']) 
    df_temp.columns = [col.replace('fuel_type_', "FUEL-binenc") for col in df_temp.columns] 

    df = pd.concat([df, df_temp], axis = 1)
    df = df.drop(columns = 'fuel_type')

    return df

def ENCODE_model(df):

    df['model']     = df['model'].str.upper()
    
    encoder         = ce.BinaryEncoder(cols='model',return_df=True)
    df_temp         = encoder.fit_transform(df['model']) 
    df_temp.columns = [col.replace('model_', "MODEL-binenc") for col in df_temp.columns] 

    df = pd.concat([df, df_temp], axis = 1)
    df = df.drop(columns = 'model')

    return df

def EXTRACT_features(df):

    df['features'] = df['features'].astype(str).str.upper()
    
    df.loc[(df['features'].str.contains('TURBOCHARGED')) | (df['features'].str.contains('SUPERCHARGED')), 'FEAT-charged_engine'] = 1
    df['FEAT-charged_engine'] = df['FEAT-charged_engine'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('SRS'), 'FEAT-srs_airbag'] = 1
    df['FEAT-srs_airbag'] = df['FEAT-srs_airbag'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('KEYLESS'), 'FEAT-keyless_ss'] = 1
    df['FEAT-keyless_ss'] = df['FEAT-keyless_ss'].fillna(0).astype(int)

    df = df.drop(columns = 'features')
    
    return df

def CALC_currentage(df):
    
    df['AGE-current'] = datetime.datetime.now().year - df['manufactured'].astype(int)
    
    return(df)

def CALC_remainage(df):

    df['title'] = df['title'].str.upper()

    df.loc[df['title'].str.contains('COE'), 'coe_temp'] = df['title'].str.split(' ').str[-1]
    df['coe_temp1']         = df['coe_temp'].str.replace(')','')
    df['coe_temp1']         = pd.to_datetime(df['coe_temp1'], format = "%m/%Y",errors='coerce')
    df['coe_temp2']         = df[['original_reg_date', 'reg_date']].max(axis = 1) + pd.offsets.DateOffset(years=10)
    df['coe_enddate']       = df['coe_temp1'].fillna(df['coe_temp2'])
    df['AGE-remaining']     = df.apply(lambda x: dateutil.relativedelta.relativedelta(x['coe_enddate'], datetime.datetime.now()).years, axis=1)
    df                      = df.drop(columns = ["coe_temp", "coe_temp1", "coe_temp2", "coe_enddate"])

    return df

def IMPUTENULL_power(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',]
    df_temp = df[related_cols + ['power']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['power'] = df['power'].fillna(df_temp[0])

    return df

def IMPUTENULL_curbweight(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',]
    df_temp = df[related_cols + ['curb_weight']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['curb_weight'] = df['curb_weight'].fillna(df_temp[0])

    return df

def IMPUTENULL_enginecap(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',]
    df_temp = df[related_cols + ['engine_cap']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['engine_cap'] = df['engine_cap'].fillna(df_temp[0])

    return df

def IMPUTENULL_depreciation(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',
                     'AGE-current',
                     'AGE-remaining'
                    ]
    df_temp = df[related_cols + ['depreciation']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['depreciation'] = df['depreciation'].fillna(df_temp[0])

    return df

def IMPUTENULL_arf(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',
                     'AGE-current',
                     'AGE-remaining'
                    ]
    df_temp = df[related_cols + ['arf']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['arf'] = df['arf'].fillna(df_temp[0])

    return df

def IMPUTENULL_omv(df):
    related_cols =  ['TRANSMISSION-manual',
                     'TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',
                     'AGE-current',
                     'AGE-remaining'
                    ]
    df_temp = df[related_cols + ['omv']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['omv'] = df['omv'].fillna(df_temp[0])

    return df

def IMPUTENULL_mileage(df):
    related_cols =  ['TRANSMISSION-manual',
                     'AGE-current',
                     'AGE-remaining'
                    ]
    df_temp = df[related_cols + ['mileage']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['mileage'] = df['mileage'].fillna(df_temp[0])

    return df

def IMPUTENULL_roadtax(df):
    related_cols =  ['TYPE-binenc0',
                     'TYPE-binenc1',
                     'TYPE-binenc2',
                     'TYPE-binenc3',
                     'MAKE-binenc0',
                     'MAKE-binenc1',
                     'MAKE-binenc2',
                     'MAKE-binenc3',
                     'MAKE-binenc4',
                     'MAKE-binenc5',
                     'MAKE-binenc6',
                     'MODEL-binenc0',
                     'MODEL-binenc1',
                     'MODEL-binenc2',
                     'MODEL-binenc3',
                     'MODEL-binenc4',
                     'MODEL-binenc5',
                     'MODEL-binenc6',
                     'MODEL-binenc7',
                     'MODEL-binenc8',
                     'MODEL-binenc9',
                     'AGE-current',
                     'AGE-remaining'
                    ]
    df_temp = df[related_cols + ['road_tax']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['road_tax'] = df['road_tax'].fillna(df_temp[0])

    return df

def RATIO_ownertoage(df):
    df['OWNER-to-AGEcurrent'] = df['no_of_owners'] / df['AGE-current']
    df.loc[df['OWNER-to-AGEcurrent'] == np.inf, 'OWNER-to-AGEcurrent'] = 1
    return df

def RATIO_miletoage(df):
    df['MILEAGE-to-AGEcurrent'] = df['mileage'] / df['AGE-current']
    df.loc[df['MILEAGE-to-AGEcurrent'] == np.inf, 'MILEAGE-to-AGEcurrent'] = df['mileage']
    return df

def RATIO_curbtopower(df):
    df['CURB-to-POWER'] = df['curb_weight'] / df['power']
    df.loc[df['CURB-to-POWER'] == np.inf, 'CURB-to-POWER'] = df[~df['CURB-to-POWER'].isnull()]['CURB-to-POWER'].min()
    return(df)
    
def RATIO_indirectcost(df):    
    df['INDIRECTcost'] = df['AGE-remaining'] * (df['road_tax'] + df['depreciation']) - df['dereg_value']
    return(df)

feature_list = ['curb_weight',
                'power',
                'engine_cap',
                'no_of_owners',
                'depreciation',
                'coe',
                'road_tax',
                'dereg_value',
                'mileage',
                'omv',
                'arf',
                'opc_scheme',
                'CAT-opccar',
                'CAT-hybridcars',
                'CAT-lowmileagecar',
                'CAT-directownersale',
                'CAT-consignmentcar',
                'CAT-importedusedvehicle',
                'CAT-electriccars',
                'CAT-premiumadcar',
                'CAT-coecar',
                'CAT-parfcar',
                'CAT-sgcarmartwarrantycars',
                'CAT-staevaluatedcar',
                'CAT-almostnewcar',
                'CAT-rare&exotic',
                'CAT-vintagecars',
                'TRANSMISSION-manual',
                'TYPE-binenc0',
                'TYPE-binenc1',
                'TYPE-binenc2',
                'TYPE-binenc3',
                'MAKE-binenc0',
                'MAKE-binenc1',
                'MAKE-binenc2',
                'MAKE-binenc3',
                'MAKE-binenc4',
                'MAKE-binenc5',
                'MAKE-binenc6',
                'FUEL-binenc0',
                'FUEL-binenc1',
                'FUEL-binenc2',
                'MODEL-binenc0',
                'MODEL-binenc1',
                'MODEL-binenc2',
                'MODEL-binenc3',
                'MODEL-binenc4',
                'MODEL-binenc5',
                'MODEL-binenc6',
                'MODEL-binenc7',
                'MODEL-binenc8',
                'MODEL-binenc9',
                'FEAT-charged_engine',
                'FEAT-srs_airbag',
                'FEAT-keyless_ss',
                'AGE-current',
                'AGE-remaining',
                'OWNER-to-AGEcurrent',
                'MILEAGE-to-AGEcurrent',
                'CURB-to-POWER',
                'INDIRECTcost']

target = 'price'

def model_preprocessing(input_df):
    
    input_df = input_df.drop(columns = ['eco_category', # same value for all records
                                        'indicative_price', # same value for all records
                                        'lifespan', #more than 90% of records null values 
                                       ])
    
    ####################### data processing ######################
    #####################################################################
    
    input_df = DEDUP_listingid(df = input_df)
    
    input_df = FIXNULL_make(df = input_df)
    input_df = FIXNULL_manufacture(df = input_df)
    input_df = FIXNULL_ownercount(df = input_df)
    input_df = FIXNULL_dereg(df = input_df)
    
    input_df = ENCODE_category(df = input_df)
    input_df = ENCODE_transmission(df = input_df)
    input_df = ENCODE_vehtype(df=input_df)
    input_df = ENCODE_make(df=input_df)
    input_df = ENCODE_fueltype(df=input_df)
    input_df = ENCODE_model(df=input_df)
    input_df = ENCODE_opcscheme(df=input_df)
    
    input_df = EXTRACT_features(df = input_df)
    
    input_df = CALC_currentage(df = input_df)
    input_df = CALC_remainage(df=input_df)
    
    input_df = IMPUTENULL_power(df=input_df)
    input_df = IMPUTENULL_curbweight(df=input_df)
    input_df = IMPUTENULL_enginecap(df=input_df)
    input_df = IMPUTENULL_depreciation(df=input_df)
    input_df = IMPUTENULL_omv(df=input_df)
    input_df = IMPUTENULL_arf(df=input_df)
    input_df = IMPUTENULL_mileage(df=input_df)
    input_df = IMPUTENULL_roadtax(df=input_df)
    
    input_df = RATIO_ownertoage(df=input_df)
    input_df = RATIO_miletoage(df=input_df)
    input_df = RATIO_curbtopower(df=input_df)
    input_df = RATIO_indirectcost(df=input_df)

    return input_df

train_df = pd.read_csv(r"./data/train.csv")
test_df = pd.read_csv(r"./data/test.csv")

####################### basic data definitions ######################
#####################################################################
# listing_id          - the unique id for the listing of a used car OK
# title               - title of the listing; free text attribute typically containing the make, model, engine type/size
# make                - make/brand of the car OK
# model               - model of the car (for the given make/brand) NOT SURE
# description         - title of the listing; free text attribute NOT SURE
# manufactured        - Year the car was manufactured OK
# original_reg_date   - Original registration data; date format string 
# reg_date            - Registration data; date format string
# type_of_vehicle     - Type of vehicle (e.g., "sports car", "luxury sedan")
# category            - Category of the car; list of categories, comma-separated
# transmission        - Type of transmission ("auto" or "manual")
# curb_weight         - Weight in kg of the vehicle without any passenger or items
# power               - Power of engine in kW
# fuel_type           - Fuel type (e.g, "petrol", "diesel", "electric")
# engine_cap          - Displacement of engine in cc (cubic centimeter)
# no_of_owners        - Number of previous owners (>=1 since all cars are used)
# depreciation        - Annual depreciation in SGD is the amount the owner loses on the value of the vehicle per year based on the assumption that the vehicle is deregistered only at the end of its 10-yr COE life-span
# coe                 - Certificate of entitlement value in SGD when first registered
# road_tax            - Road tax value in SGD calculated based on the engine capacity on a per annum basis
# dereg_value         - deregistration value in SGD one will get back from the government upon deregistering the vehicle for use in Singapore
# mileage             - Number of kilometers driven
# omv                 - Open Market Value in SGD assessed by the Singapore Customs
# arf                 - Additional Registration Fee in SGD is a tax imposed upon registration of a vehicle
# opc_scheme          - Off-peak car scheme
# lifespan            - Date stated on the lifespan marks the day that the vehicle must be deregistered
# eco_category        - Eco category of vehicle
# features            - Noteworthy features; free text attribute
# accessories         - Noteworthy accessories ; free text attribute
# indicative_price    - General guide to the price in SGD of the vehicle
# price               - Resale price in SGD of the car

train_df = model_preprocessing(input_df = train_df)
test_df = model_preprocessing(input_df = test_df)

####################### hyperparameters tuning ######################
#####################################################################

# Define the hyperparameter distributions
param_dist = {
            'learning_rate'     : stats.uniform(0.01, 0.1),
            'gamma'             : stats.uniform(0.01, 1.0),    
            'alpha'             : stats.uniform(0.01, 0.1),     
            
            'max_depth'         : stats.randint(3, 10),
            'subsample'         : stats.uniform(0.2, 0.8),
            'colsample_bytree'  : stats.uniform(0.2, 0.8),
            'colsample_bylevel' : stats.uniform(0.2, 0.8),
            'min_child_weight'  : stats.randint(3, 88),   
            }

xgb_model = xgboost.XGBRegressor(random_state = 888888, 
                                 n_estimators = 1888,)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(xgb_model, 
                                   param_distributions  = param_dist, 
                                   n_iter               = 88, 
                                   cv                   = 5, 
                                   random_state         = 888888,
                                   scoring              = 'neg_root_mean_squared_error', 
                                  verbose               = 2)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(train_df[feature_list], train_df[target])
random_search_df = pd.DataFrame(random_search.cv_results_)
random_search_df = random_search_df[[ 'params',
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

####################### model training ######################
#############################################################
print(random_search_df['objfunc'].iloc[0])
best_params = random_search_df['params'].iloc[0]
# {'alpha': 0.0958268019718619,
# 'colsample_bylevel': 0.6953798421848867,
# 'colsample_bytree': 0.938505748984886,
# 'gamma': 0.5343798406380182,
# 'learning_rate': 0.036462517542717766,
# 'max_depth': 7,
# 'min_child_weight': 42,
# 'subsample': 0.9101930852435802}

xbg_model = xgboost.XGBRegressor(random_state           = 888888,
                                 n_estimators           = 1888,
                             
                                 learning_rate          = best_params['learning_rate'], #  step size shrinkage used in updates to prevent overfitting. Lower values make the model more robust by taking smaller steps.
                                 gamma                  = best_params['gamma'], #minimum loss reduction required to make a further partition on a leaf node of the tree. Higher values increase the regularization.
                                 alpha                  = best_params['alpha'], #L2 regularization term on weights. 
                                 
                                 max_depth              = best_params['max_depth'], #maximum depth of a tree
                                 subsample              = best_params['subsample'], #percentage of rows used for each tree construction. 
                                 colsample_bytree       = best_params['colsample_bytree'], #percentage of columns used for each tree construction
                                 colsample_bylevel      = best_params['colsample_bylevel'], #is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
                                 min_child_weight       = best_params['min_child_weight'], #Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                                verbosity = 2)

xbg_model.fit(train_df[feature_list], train_df[target])
submission_df = pd.DataFrame(xbg_model.predict(test_df[feature_list])).reset_index().rename(columns = {'index':"Id", 0:'Predicted'})
submission_df['Predicted'] = round(submission_df['Predicted'])
submission_df
