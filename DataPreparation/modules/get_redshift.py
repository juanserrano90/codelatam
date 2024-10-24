# -- Authors: Paula Gálvez Molina
# Last Modified: Oct. 24th, 2024

import pandas as pd 

def get_redshift(sn_name, csv_file='/home/paulagm/GitRepos/codelatam/Data/Redshift/sn_redshift.csv'):
    # --Check if sn_name is a string
    if not isinstance(sn_name, str):
        raise TypeError(f"Error: Expected 'sn_name' to be a string, but got {type(sn_name).__name__}")
    
    # --Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {csv_file} was not found.")
    
    # --Check if the DataFrame contains the 'SN Name' column and if it's set as the index
    if 'SN Name' not in df.columns:
        raise ValueError("Error: 'SN Name' column is not found in the CSV file.")
    
    # --Set 'SN Name' as index if not already set
    if df.index.name != 'SN Name':
        df.set_index('SN Name', inplace=True)
    
    # --Check if the sn_name exists in the index
    if sn_name not in df.index:
        raise ValueError(f"Error: Redshift value for '{sn_name}' not found.")
    
    redshift = df.loc[sn_name, 'redshift']
    # Return the redshift value
    # logging.info('------Redshift: {redshift}]')
    return redshift


    return spiked