import pandas as pd
import scipy as sp
import xgboost as xgb
import pprint as pp
import common_helper


def clean(df, drop_collinearity = True, inplace = False):
    # convert numeric to string
    variables = ['v38', 'v62', 'v72', 'v129']
    df = common_helper.numeric_to_categorical(df, variables)
    
    # drop column v22
    df = df.drop('v22', axis=1, inplace = inplace)
    
    # Drop highly correlated columns
    if(drop_collinearity):
        df = drop_highly_correlated_variables(df, inplace)
        
    return df
    
    
def drop_highly_correlated_variables(df, inplace = False):
    '''
    @summary: Check highly correlated column pairs. Drop the one in a pair with higher missingness.
    If one varialbe is marked to be dropped, other pairs contain it will be skipped for checking.
    '''
    corr = common_helper.get_corr(df, 0.9)
    vars_to_drop = []

    for var_corr in corr:
        if(var_corr['_var1'] not in vars_to_drop and var_corr['var2_na'] not in vars_to_drop):
            var_to_drop = var_corr['_var1'] if var_corr['var1_na'] > var_corr['var2_na'] else var_corr['_var2']
            vars_to_drop.append(var_to_drop)

    return df.drop(vars_to_drop, axis=1, inplace = inplace)
    

def get_categorical_variables(df):
    filter_categorical = df.dtypes == 'object'
    vars_cate = df.columns[filter_categorical]
    
    return vars_cate


def get_numeric_variables(df):
    filter_categorical = df.dtypes == 'object'
    vars_numeric = df.columns[-filter_categorical]
    
    return vars_numeric
        
def impute_cate_with_na_numeric_with_interpolate_linear(df):
    df_predictors = df.iloc[:,2:]
    df_target = df.iloc[:,:2]
    
    # Find categorical variables
    df_cate = df_predictors[get_categorical_variables(df_predictors)]
    
    # Impute categorical data, fill in with 'NA'
    df_cate = df_cate.fillna('NA')
    
    # Find umeric variables
    df_numeric = df_predictors[get_numeric_variables(df_predictors)]
    
    # Impute numeric variables with interpolate linear
    df_numeric = df_numeric.interpolate(method='linear')

    return pd.concat([df_target, df_cate, df_numeric], axis = 1)


def impute_cate_with_na_numeric_with_outlier(df):    
    df_predictors = df.iloc[:,2:]
    df_target = df.iloc[:,:2]
    
    # Find categorical variables
    vars_cate = get_categorical_variables(df_predictors)
    df_cate = df_predictors[vars_cate]
    
    # Impute categorical data, fill in with 'NA'
    df_cate = df_cate.fillna('NA')
    
    # Find umeric variables
    df_numeric = df_predictors[get_categorical_variables(df_predictors)]
    
    # Impute numeric variables with interpolate linear
    df_numeric = df_numeric.fillna(-999)

    return pd.concat([df_target, df_cate, df_numeric], axis = 1)

