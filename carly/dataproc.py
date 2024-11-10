import json
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

with open("carly/fuel_type_map.json", "r") as f: 
    FUEL_TYPE_DICT = json.load(f)

def DEDUP_listingid(df):
    
    """
    
    Deduplicates rows in a DataFrame based on 'listing_id' after ensuring all IDs are strings.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'listing_id' column.

    Returns:
    pd.DataFrame: DataFrame with unique 'listing_id' values.
    
    """
    
    df['listing_id']    = df['listing_id'].astype(int).astype(str).str.strip()  # Standardize listing_id as integer then string
    df                  = df.drop_duplicates('listing_id')  # Remove duplicate rows based on listing_id
    
    return df


def FIXNULL_make(df): 
    
    """
    
    Standardizes and fills missing 'make' column values by analyzing 'title'.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'make' and 'title' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with filled 'make' values.
    
    """
    
    df['make']          = df['make'].str.upper().str.replace(' ', '').str.strip()  # Standardize make format
    df['title']         = df['title'].str.upper()  # Standardize title format

    make_list           = [make for make in df['make'].unique() if isinstance(make, str)]  # List of unique make strings
    df['make_temp']     = df['title'].str.split().str[0]  # Assume first word in title represents make
    
    # Match 'make_temp' with closest match in make_list, use it if 'make' is missing
    df['make_temp']     = df.apply(lambda x: difflib.get_close_matches(x['make_temp'], make_list, n=1)[0], axis=1)
    df['make']          = df['make'].fillna(df['make_temp'])  # Fill missing 'make' values
    df                  = df.drop(columns='make_temp')  # Clean up temporary column

    return df


def FIXNULL_manufacture(df):
    
    """
    
    Estimates 'manufactured' year from registration dates if missing.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'original_reg_date' and 'reg_date' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with filled 'manufactured' values.
    
    """
    
    # Parse registration dates
    df['original_reg_date']         = pd.to_datetime(df['original_reg_date'], format="%d-%b-%Y")
    df['reg_date']                  = pd.to_datetime(df['reg_date'], format="%d-%b-%Y")

    df['original_reg_date_temp']    = df['original_reg_date'].dt.year  # Extract year from original_reg_date
    df['reg_date_temp']             = df['reg_date'].dt.year  # Extract year from reg_date
    # Fill 'manufactured' with the earlier registration year
    df['manufactured']              = df['manufactured'].fillna(df[['original_reg_date_temp', 'reg_date_temp']].min(axis=1))
    df['manufactured']              = df['manufactured'].astype(int).astype(str)  # Convert to integer string for consistency
    df                              = df.drop(columns=['original_reg_date_temp', 'reg_date_temp'])  # Remove temporary columns

    return df


def FIXNULL_ownercount(df):
    
    """
    
    Fills missing 'no_of_owners' values with 1.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'no_of_owners' column.

    Returns:
    pd.DataFrame: Updated DataFrame with filled 'no_of_owners'.
    
    """
    
    df['no_of_owners'] = df['no_of_owners'].fillna(1)  # Fill missing values with 1 (assumed default)
    
    return df


def FIXNULL_dereg(df):
    
    """
    Fills missing 'dereg_value' with 0.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'dereg_value' column.

    Returns:
    pd.DataFrame: Updated DataFrame with filled 'dereg_value'.
    
    """
    
    df['dereg_value'] = df['dereg_value'].fillna(0)  # Replace NaN with 0
    
    return df

def FIXNULL_fueltype(df):
    """
    Fills missing 'fuel_type' from existing data points based on make and model.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'fuel_type' column.

    Returns:
    pd.DataFrame: Updated DataFrame with filled 'fuel_type'.
    """
    # Extract fuel type from category
    fuel_keywords = {
        'electric': 'electric',
        'hybrid': 'petrol-electric'
    }

    def extract_fuel_type(category_text):
        category_text = category_text.lower()  
        for keyword, fuel_type in fuel_keywords.items():
            if keyword in category_text:
                return fuel_type
        return None
    
    df['fuel_type_category_fill'] = df['fuel_type']

    df.loc[df['fuel_type'].isna(), 'fuel_type_category_fill'] = df['category'].apply(extract_fuel_type)

    # Define a function to fill missing fuel types
    def fill_fuel_type(row):
        if pd.isna(row['fuel_type']):
            return FUEL_TYPE_DICT.get(f"{row['make']}_{row['model']}", None)
        return row['fuel_type']

    # Apply the function to fill in missing values
    df['fuel_type_model_make_fill'] = df['fuel_type']
    df.loc[df['fuel_type'].isna(), 'fuel_type_model_make_fill']  = df.apply(fill_fuel_type, axis=1)

    df['fuel_type'] = df['fuel_type_model_make_fill']
    df = df.drop(columns=['fuel_type_model_make_fill', 'fuel_type_category_fill'])

    return df

def ENCODE_category(df, drop=True):
    
    """
    
    Expands and encodes 'category' column into individual category flags.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'category' column.
    drop (bool): Whether to drop the original 'category' column after encoding.

    Returns:
    pd.DataFrame: DataFrame with new categorical columns.
    
    """
    
    # Split category column, expand into separate columns
    df_temp                 = df['category'].str.split(',', expand=True).astype(str)
    # Get unique set of categories from expanded columns
    category_set            = {category.strip() for category in sum(df['category'].str.split(',').tolist(), []) if category.strip()}

    # Create binary columns for each category in the set
    category_dflist         = [df_temp.apply(lambda x: sum(x.str.contains(category)), axis=1).rename(f"CAT-{category.replace(' ','')}") for category in category_set]
    df                      = pd.concat([df] + category_dflist, axis=1)  # Append new columns to df
    df                      = df.drop(columns=['category']) if drop else df

    # Recode specific categories into meaningful labels
    df['CAT-coeparf']       = df['CAT-coecar'].astype(str).str.replace('1', 'COE').str.replace('0', 'PARF')
    df['CAT-elechybrid']    = df['CAT-hybridcars'].astype(str).str.replace('1', 'HYBRID').str.replace('0', 'ELEC')

    if drop:
        # Drop the specific redundant columns after creating new labels
        df                  = df.drop(columns=['CAT-electriccars', 'CAT-hybridcars', 'CAT-coecar', 'CAT-parfcar'])

    return df


def ENCODE_transmission(df, drop=True):
    
    """
    
    Encodes 'transmission' column as binary.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'transmission' column.
    drop (bool): Whether to drop the original 'transmission' column after encoding.

    Returns:
    pd.DataFrame: Updated DataFrame with binary transmission column.
    
    """
    
    # Create binary indicator column for 'transmission' (manual or not)
    df['TRANSMISSION-manual'] = pd.get_dummies(df['transmission'], drop_first=True, dtype=int)
    
    if drop:
        df = df.drop(columns=['transmission'])  # Drop original transmission column if drop=True

    return df

def ENCODE_vehtype(df, drop=True):
    
    """
    
    Encodes 'type_of_vehicle' column using binary encoding.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'type_of_vehicle' column.
    drop (bool): Whether to drop the original 'type_of_vehicle' column after encoding.

    Returns:
    pd.DataFrame: DataFrame with binary-encoded vehicle type columns.
    
    """
    
    # Standardize format: convert to uppercase, remove spaces
    df['type_of_vehicle']   = df['type_of_vehicle'].str.upper().str.strip().str.replace(' ', '')

    # Initialize binary encoder and transform 'type_of_vehicle' column
    encoder                 = ce.BinaryEncoder(cols='type_of_vehicle', return_df=True)
    df_temp                 = encoder.fit_transform(df['type_of_vehicle']) 

    # Rename columns to a more descriptive format
    df_temp.columns         = [col.replace('type_of_vehicle_', "TYPE-binenc") for col in df_temp.columns]
    
    # Concatenate new columns to original DataFrame
    df                      = pd.concat([df, df_temp], axis=1)
    
    if drop:
        df = df.drop(columns='type_of_vehicle')  # Drop original column if drop=True

    return df


def ENCODE_make(df, drop=True):
    
    """
    
    Encodes 'make' column using binary encoding.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'make' column.
    drop (bool): Whether to drop the original 'make' column after encoding.

    Returns:
    pd.DataFrame: DataFrame with binary-encoded make columns.
    
    """
    
    # Initialize binary encoder and transform 'make' column
    encoder             = ce.BinaryEncoder(cols='make', return_df=True)
    df_temp             = encoder.fit_transform(df['make'])

    # Rename columns to a more descriptive format
    df_temp.columns     = [col.replace('make_', "MAKE-binenc") for col in df_temp.columns]

    # Concatenate new columns to original DataFrame
    df                  = pd.concat([df, df_temp], axis=1)
    
    if drop:
        df = df.drop(columns=['make'])  # Drop original column if drop=True

    return df


def ENCODE_opcscheme(df):
    
    """
    
    Encodes 'opc_scheme' as binary, setting non-null values to 1, and nulls to 0.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'opc_scheme' column.

    Returns:
    pd.DataFrame: DataFrame with binary 'opc_scheme' column.
    
    """
    
    # Assign binary values based on presence of data
    df.loc[~(df['opc_scheme'].isnull()), 'opc_scheme'] = 1
    df['opc_scheme'] = df['opc_scheme'].fillna(0)  # Fill remaining nulls with 0

    return df


def ENCODE_fueltype(df, drop=True):
    
    """
    
    Encodes 'fuel_type' column using binary encoding, replacing missing values with 'unknown'.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'fuel_type' column.
    drop (bool): Whether to drop the original 'fuel_type' column after encoding.

    Returns:
    pd.DataFrame: DataFrame with binary-encoded fuel type columns.
    
    """
    
    # Fill null values, standardize format
    df['fuel_type'] = df['fuel_type'].fillna('unknown').str.upper().str.strip()

    # Initialize binary encoder and transform 'fuel_type' column
    encoder         = ce.BinaryEncoder(cols='fuel_type', return_df=True)
    df_temp         = encoder.fit_transform(df['fuel_type'])

    # Rename columns to a more descriptive format
    df_temp.columns = [col.replace('fuel_type_', "FUEL-binenc") for col in df_temp.columns]

    # Concatenate new columns to original DataFrame
    df              = pd.concat([df, df_temp], axis=1)

    if drop:
        df = df.drop(columns='fuel_type')  # Drop original column if drop=True

    return df

def ENCODE_model(df, drop=True):
    
    """
    
    Encodes 'model' column using binary encoding.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'model' column.
    drop (bool): Whether to drop the original 'model' column after encoding.

    Returns:
    pd.DataFrame: DataFrame with binary-encoded model columns.
    
    """
    
    df['model']     = df['model'].str.upper()  # Standardize format to uppercase

    # Initialize binary encoder and transform 'model' column
    encoder         = ce.BinaryEncoder(cols='model', return_df=True)
    df_temp         = encoder.fit_transform(df['model'])

    # Rename columns to a more descriptive format
    df_temp.columns = [col.replace('model_', "MODEL-binenc") for col in df_temp.columns]

    # Concatenate new columns to original DataFrame
    df              = pd.concat([df, df_temp], axis=1)

    if drop:
        df = df.drop(columns='model')  # Drop original column if drop=True

    return df

def EXTRACT_features(df):
    
    """
    
    Extracts detailed vehicle features from 'features' column and creates specific feature flags.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'features' column.

    Returns:
    pd.DataFrame: DataFrame with extracted feature columns and flags.
    
    """
    
    df['features']      = df['features'].astype(str).str.upper()  # Standardize features format

    # Define feature flags based on keywords in 'features'
    df.loc[(df['features'].str.contains('4 CYLINDER')) | (df['features'].str.contains('4-CYLINDER')) | (df['features'].str.contains('FOUR CYLINDER')), 'FEAT-CYLcount'] = 4
    df.loc[(df['features'].str.contains('5 CYLINDER')) | (df['features'].str.contains('5-CYLINDER')) | (df['features'].str.contains('FIVE CYLINDER')), 'FEAT-CYLcount'] = 5
    df.loc[(df['features'].str.contains('6 CYLINDER')) | (df['features'].str.contains('6-CYLINDER')) | (df['features'].str.contains('SIX CYLINDER')), 'FEAT-CYLcount'] = 6
    df['FEAT-CYLcount'] = df['FEAT-CYLcount'].fillna(0).astype(int)  # Default to 0 if no cylinder info found

    # Extract numerical values for specific features
    df['FEAT-brakehp']  = df['features'].str.extract(r'(\d+)(?:BHP| BHP)') 
    df['FEAT-torque']   = df['features'].str.extract(r'(\d+)(?:NM| NM)') 
    df['FEAT-valve']    = df['features'].str.extract(r'(\d+)(?:VALVE| VALVE)') 

    # Define binary flags for additional features
    df.loc[(df['features'].str.contains('TURBOCHARGED')) | (df['features'].str.contains('SUPERCHARGED')), 'FEAT-charged_engine'] = 1
    df['FEAT-charged_engine']   = df['FEAT-charged_engine'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('SRS'), 'FEAT-srs_airbag'] = 1
    df['FEAT-srs_airbag']       = df['FEAT-srs_airbag'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('KEYLESS'), 'FEAT-keyless_ss'] = 1
    df['FEAT-keyless_ss']       = df['FEAT-keyless_ss'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('GUARANTEE'), 'FEAT-guarantee'] = 1
    df['FEAT-guarantee']        = df['FEAT-guarantee'].fillna(0).astype(int)

    # Define drive-type feature flags based on keywords
    df.loc[df['features'].str.contains('AWD'), 'FEAT-awd'] = 1
    df['FEAT-awd'] = df['FEAT-awd'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('4WD'), 'FEAT-4wd'] = 1
    df['FEAT-4wd'] = df['FEAT-4wd'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('RWD'), 'FEAT-rwd'] = 1
    df['FEAT-rwd'] = df['FEAT-rwd'].fillna(0).astype(int)

    df.loc[df['features'].str.contains('FWD'), 'FEAT-fwd'] = 1
    df['FEAT-fwd'] = df['FEAT-fwd'].fillna(0).astype(int)

    # Drop the original 'features' column after extraction
    df = df.drop(columns='features')
    
    return df

def EXTRACT_desc(df): 
    
    """
    
    Extracts key phrases from the 'description' column and creates binary flags based on their presence.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'description' column.

    Returns:
    pd.DataFrame: DataFrame with extracted description-based flags.
    
    """
    
    df['description'] = df['description'].astype(str).str.upper()  # Standardize description format

    # Define binary flags based on keywords in 'description'
    df.loc[df['description'].str.contains('LOAN'), 'DESC-loan'] = 1
    df['DESC-loan']         = df['DESC-loan'].fillna(0).astype(int)

    df.loc[df['description'].str.contains('MAINTAINED'), 'DESC-maintained'] = 1
    df['DESC-maintained']   = df['DESC-maintained'].fillna(0).astype(int)

    df.loc[df['description'].str.contains('ACCIDENT FREE'), 'DESC-accifree'] = 1
    df['DESC-accifree']     = df['DESC-accifree'].fillna(0).astype(int)

    df.loc[df['description'].str.contains('WARRANTY'), 'DESC-warranty'] = 1
    df['DESC-warranty']     = df['DESC-warranty'].fillna(0).astype(int)

    df.loc[df['description'].str.contains('TRADE IN'), 'DESC-tradein'] = 1
    df['DESC-tradein']      = df['DESC-tradein'].fillna(0).astype(int)

    df = df.drop(columns='description')  # Remove original description column after extraction

    return df


def EXTRACT_accessories(df): 
    
    """
    
    Extracts key accessories from 'accessories' column and creates binary flags based on their presence.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'accessories' column.

    Returns:
    pd.DataFrame: DataFrame with extracted accessories-based flags.
    
    """
    
    df['accessories'] = df['accessories'].astype(str).str.upper()  # Standardize accessories format

    # Define binary flags based on keywords in 'accessories'
    df.loc[df['accessories'].str.contains('INSPECTION'), 'ACCESS-inspection'] = 1
    df['ACCESS-inspection']     = df['ACCESS-inspection'].fillna(0).astype(int)

    df.loc[df['accessories'].str.contains('AUDIO'), 'ACCESS-audio'] = 1
    df['ACCESS-audio']          = df['ACCESS-audio'].fillna(0).astype(int)

    df.loc[df['accessories'].str.contains('CAMERA'), 'ACCESS-camera'] = 1
    df['ACCESS-camera']         = df['ACCESS-camera'].fillna(0).astype(int)

    df.loc[(df['accessories'].str.contains('SPORT RIM')) | (df['accessories'].str.contains('SPORTS RIM')), 'ACCESS-sportrim'] = 1
    df['ACCESS-sportrim']       = df['ACCESS-sportrim'].fillna(0).astype(int)

    df = df.drop(columns='accessories')  # Remove original accessories column after extraction

    return df

def CALC_currentage(df):

    """

    Calculates vehicle age from manufacturing date and registration date.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'manufactured', 'original_reg_date', and 'reg_date' columns.

    Returns:
    pd.DataFrame: DataFrame with calculated current age columns.

    """

    # Calculate age from manufacturing year
    df['AGE-currentMANU']   = datetime.datetime.now().year - df['manufactured'].astype(int)

    # Use 'original_reg_date' if available; otherwise, use 'reg_date'
    df['reg_date_temp']     = df['original_reg_date'].fillna(df['reg_date'])
    # Calculate age from registration date in months
    df['AGE-currentREG']    = df.apply(lambda x: dateutil.relativedelta.relativedelta(x['reg_date_temp'], datetime.datetime.now()).months, axis=1)

    df = df.drop(columns=['reg_date_temp'])  # Drop temporary column after use

    return df


def CALC_remainage(df):
    
    """
    
    Calculates remaining age based on COE scheme or a 10-year expiration from registration date.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'title', 'original_reg_date', and 'reg_date' columns.

    Returns:
    pd.DataFrame: DataFrame with calculated remaining age columns.
    
    """
    
    df['title']         = df['title'].str.upper()  # Standardize title format

    # Extract COE expiry date from title if available
    df.loc[df['title'].str.contains('COE'), 'coe_temp'] = df['title'].str.split(' ').str[-1]
    df['coe_temp1']     = df['coe_temp'].str.replace(')', '')  # Clean formatting
    df['coe_temp1']     = pd.to_datetime(df['coe_temp1'], format="%m/%Y", errors='coerce')

    # If COE not available, assume 10 years from registration date
    df['coe_temp2']     = df[['original_reg_date', 'reg_date']].max(axis=1) + pd.offsets.DateOffset(years=10)
    df['coe_enddate']   = df['coe_temp1'].fillna(df['coe_temp2'])

    # Calculate years remaining until COE end date
    df['AGE-remaining'] = df.apply(lambda x: dateutil.relativedelta.relativedelta(x['coe_enddate'], datetime.datetime.now()).years, axis=1)
    
    df = df.drop(columns=["coe_temp", "coe_temp1", "coe_temp2", "coe_enddate"])  # Remove temporary columns

    return df

def IMPUTENULL_power(df):
    
    """
    
    Imputes missing 'power' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'power' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'power' values.
    
    """
    
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
    
    df_temp         = df[related_cols + ['power']]  # Data subset for imputation
    imputer         = KNNImputer(n_neighbors=5, weights='distance')  # Initialize KNN imputer
    df_temp         = imputer.fit_transform(df_temp)  # Perform imputation
    df['power']     = df['power'].fillna(pd.Series(df_temp[:, -1]))  # Fill NaNs with imputed values

    return df


def IMPUTENULL_curbweight(df):
    
    """
    
    Imputes missing 'curb_weight' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'curb_weight' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'curb_weight' values.
    
    """
    
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
    
    df_temp             = df[related_cols + ['curb_weight']]  # Data subset for imputation
    imputer             = KNNImputer(n_neighbors=5, weights='distance')
    df_temp             = imputer.fit_transform(df_temp)
    df['curb_weight']   = df['curb_weight'].fillna(pd.Series(df_temp[:, -1]))

    return df

def IMPUTENULL_enginecap(df):

    """
    
    Imputes missing 'engine_cap' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'engine_cap' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'engine_cap' values.
    
    """
    
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
    df['engine_cap'] = df['engine_cap'].fillna(pd.Series(df_temp[:, -1]))
    
    return df

def IMPUTENULL_depreciation(df):
    
    """
    
    Imputes missing 'depreciation' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'depreciation' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'depreciation' values.
    
    """
    
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
                     'AGE-currentMANU',            
                     'AGE-remaining'
                    ]
    
    df_temp             = df[related_cols + ['depreciation']]
    imputer             = KNNImputer(n_neighbors=5, weights='distance')
    df_temp             = imputer.fit_transform(df_temp)
    df['depreciation']  = df['depreciation'].fillna(pd.Series(df_temp[:, -1]))

    return df

def IMPUTENULL_arf(df):
    
    """
    
    Imputes missing 'arf' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'arf' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'ard' values.
    
    """    
    
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
                     'AGE-currentMANU',
                     'AGE-remaining'
                    ]
    df_temp     = df[related_cols + ['arf']]
    imputer     = KNNImputer(n_neighbors=5, weights='distance')
    df_temp     = imputer.fit_transform(df_temp)
    df['arf']   = df['arf'].fillna(pd.Series(df_temp[:, -1]))

    return df

def IMPUTENULL_omv(df):
    
    """
    
    Imputes missing 'omv' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'omv' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'omv' values.
    
    """
    
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
                     'AGE-currentMANU',
                     'AGE-remaining'
                    ]
    df_temp     = df[related_cols + ['omv']]
    imputer     = KNNImputer(n_neighbors=5, weights='distance')
    df_temp     = imputer.fit_transform(df_temp)
    df['omv']   = df['omv'].fillna(pd.Series(df_temp[:, -1]))

    return df

def IMPUTENULL_mileage(df):
    
    """
    
    Imputes missing 'mileage' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'mileage' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'mileage values.
    
    """        
    
    related_cols =  ['TRANSMISSION-manual',
                     'AGE-currentMANU',
                     'AGE-remaining'
                    ]
    
    df_temp         = df[related_cols + ['mileage']]
    imputer         = KNNImputer(n_neighbors=5, weights='distance')
    df_temp         = imputer.fit_transform(df_temp)
    df['mileage']   = df['mileage'].fillna(pd.Series(df_temp[:, -1]))

    return df

def IMPUTENULL_roadtax(df):
    
    """
    
    Imputes missing 'roadtax' values using K-Nearest Neighbors based on related columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'roadtax' and related feature columns.

    Returns:
    pd.DataFrame: DataFrame with imputed 'roadtax' values.
    
    """    
    
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
                     'AGE-currentMANU',
                     'AGE-remaining'
                    ]
    df_temp         = df[related_cols + ['road_tax']]
    imputer         = KNNImputer(n_neighbors=5, weights='distance')
    df_temp         = imputer.fit_transform(df_temp)
    df['road_tax']  = df['road_tax'].fillna(pd.Series(df_temp[:, -1]))

    return df

def RATIO_ownertoage(df):
    
    """
    
    Calculates the ratio of the number of owners to the vehicle's age.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'no_of_owners' and 'AGE-currentMANU'.

    Returns:
    pd.DataFrame: DataFrame with new 'OWNER-to-AGEcurrent' ratio column.
    
    """
    
    df['OWNER-to-AGEcurrent'] = df['no_of_owners'] / df['AGE-currentMANU']
    df.loc[df['OWNER-to-AGEcurrent'] == np.inf, 'OWNER-to-AGEcurrent'] = 1  # Handle division by zero
    
    return df

def RATIO_miletoage(df):
    
    """
    
    Calculates the ratio of mileage to vehicle's age.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'mileage' and 'AGE-currentMANU'.

    Returns:
    pd.DataFrame: DataFrame with new 'MILEAGE-to-AGEcurrent' ratio column.
    
    """
    
    df['MILEAGE-to-AGEcurrent'] = df['mileage'] / df['AGE-currentMANU']
    df.loc[df['MILEAGE-to-AGEcurrent'] == np.inf, 'MILEAGE-to-AGEcurrent'] = df['mileage']  # Handle division by zero
    
    return df

def RATIO_miletoowner(df):
    
    """
    
    Calculates the ratio of mileage to the number of owners.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'mileage' and 'no_of_owners'.

    Returns:
    pd.DataFrame: DataFrame with new 'MILEAGE-to-OWNER' ratio column.
    
    """
    
    df['MILEAGE-to-OWNER'] = df['mileage'] / df['no_of_owners']
    
    return df

def RATIO_curbtopower(df):
    
    """
    
    Calculates the ratio of curb weight to power.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'curb_weight' and 'power'.

    Returns:
    pd.DataFrame: DataFrame with new 'CURB-to-POWER' ratio column.
    
    """
    
    df['CURB-to-POWER'] = df['curb_weight'] / df['power']
    # Replace infinite values with minimum finite value in 'CURB-to-POWER'
    df.loc[df['CURB-to-POWER'] == np.inf, 'CURB-to-POWER'] = df[~df['CURB-to-POWER'].isnull()]['CURB-to-POWER'].min()
    
    return df

def RATIO_indirectcost(df):    
    
    """
    
    Calculates an indirect cost based on age, road tax, depreciation, and deregistration value.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'AGE-remaining', 'road_tax', 'depreciation', and 'dereg_value'.

    Returns:
    pd.DataFrame: DataFrame with calculated 'INDIRECTcost' column.
    
    """
    
    df['INDIRECTcost'] = df['AGE-remaining'] * (df['road_tax'] + df['depreciation']) - df['dereg_value']
    
    return df

def RATIO_ARFtoOMV(df):  
    
    """
    
    Calculates the ratio of ARF to OMV.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'arf' and 'omv'.

    Returns:
    pd.DataFrame: DataFrame with calculated 'ARF-to-OMV' ratio column.
    
    """
    
    df['ARF-to-OMV'] = df['arf'] / df['omv']
    return df

def RATIO_ARFtoDEREG(df):    
    
    """
    
    Calculates the ratio of ARF to deregistration value.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'arf' and 'dereg_value'.

    Returns:
    pd.DataFrame: DataFrame with calculated 'ARF-to-DEREG' ratio column.
    
    """
    
    df['ARF-to-DEREG'] = df['arf'] / df['dereg_value']
    df.loc[df['ARF-to-DEREG'] == np.inf, 'ARF-to-DEREG'] = 0  # Handle division by zero
    
    return df


