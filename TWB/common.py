import numpy as np
import pandas as pd

def df_to_upper(df):
    """
    Convert textual columns to uppercase
    
    parameters:
    -----------
    df: pandas.DataFrame
        The inputn data frame
        
    returns:
    pd.DataFrame
    """
    df = df.copy()
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x).upper())
        #fi
    #efor
    return df
#edef