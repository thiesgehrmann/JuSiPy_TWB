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

def freq(lst):
    """
    calculate the frequency of items in a list
    
    parameters:
    -----------
    lst: List[Hashable Objects]
        The list you have
        
    returns:
    --------
    freq: dict[hashable object -> int]
    """
    C = {}
    for i in lst:
        C[i] = C.get(i, 0) + 1
    #efor
    return C
#edef