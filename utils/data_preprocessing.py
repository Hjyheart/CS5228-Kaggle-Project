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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

def IMPUTENULL_deregvalue(df):
    related_cols = ['depreciation', 
                    'AGE-remaining', 
                    'omv', 
                    'arf', 
                    'rare'
                    ]
    df_temp = df[related_cols + ['dereg_value']]
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_temp = imputer.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp[:,-1])
    df['dereg_value'] = df['dereg_value'].fillna(df_temp[0])

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


def identify_outliers(df: pd.Dataframe):
    """
    Identify outliers with DBSCAN after scaling and decomposition. 

    Args:
        df (pd.Dataframe): Train Dataframe
    
    Returns:
        df (pd.DataFrame): Dataframe with added cluster column (cluster -1 anomalies)
    """

    df_numerical = df.select_dtypes(include=['number'])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)


    pca = PCA()
    _ = pca.fit(df_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1  # +1 because index is 0-based
    print(f"Number of components to retain 90% variance: {n_components_90}")

    pca = PCA(n_components=n_components_90)  # Use the number of components determined earlier
    pca_data = pca.fit_transform(df_scaled)

        # Use NearestNeighbors to find the k-nearest distances
    k = 5  # Common choice for DBSCAN; can adjust based on data characteristics
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(pca_data)
    distances, indices = neighbors.kneighbors(pca_data)
    # Get the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])

    # # Plot k-distance graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_distances)
    # plt.title('K-Distance Graph')
    # plt.xlabel('Data Points sorted by Distance to their {}-th Nearest Neighbor'.format(k))
    # plt.ylabel('Distance')
    # plt.grid()
    # plt.show()

    # Initialize DBSCAN to detect anomalies
    db = DBSCAN(eps=5, min_samples=5) # eps : [4, 5 , 9]
    db.fit(df_scaled)

    labels = db.labels_
    df['cluster'] = labels

    # Identify anomalies (labeled as -1 by DBSCAN)
    anomalies = df[df['cluster'] == -1]
    print(f"Anomalies found: {len(anomalies)}")

    return df

