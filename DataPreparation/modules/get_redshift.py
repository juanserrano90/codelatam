# -- Authors: Paula Gálvez Molina - Valentina Contreras Rojas
# Last Modified: Oct. 24th, 2024

import pandas as pd 
import logging


def get_redshift(sn_name, csv_file='https://raw.githubusercontent.com/juanserrano90/codelatam/main/Data/Redshift/sn_redshift.csv'):
    # -- Check if sn_name is a string
    if not isinstance(sn_name, str):
        raise TypeError(f"Error: Expected 'sn_name' to be a string, but got {type(sn_name).__name__}")

    # -- Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise FileNotFoundError(f"Error: Could not load the file {csv_file}. Details: {e}")

    # -- Print column names for debugging
    print("Columnas en el archivo CSV:", df.columns.tolist())

    # -- Check if the DataFrame contains the 'SN_Name' or 'old_name' column
    if 'SN_Name' not in df.columns and 'old_name' not in df.columns:
        raise ValueError("Error: Neither 'SN_Name' nor 'old_name' column found in the CSV file.")

    # -- Standardize SN Name and old_name columns (remove extra spaces)
    if 'SN_Name' in df.columns:
        df['SN_Name'] = df['SN_Name'].str.strip()
    if 'old_name' in df.columns:
        df['old_name'] = df['old_name'].str.strip()

    # -- Set 'SN_Name' as index first if it exists, otherwise use 'old_name'
    if 'SN_Name' in df.columns:
        df = df.set_index('SN_Name')
    else:
        df = df.set_index('old_name')

    # -- Print available SN names or old names for debugging
    print("Available SN names or old names:", list(df.index)[:10])  # Muestra solo los primeros 10 para evitar exceso de texto

    # -- Check if the sn_name exists in the index
    if sn_name not in df.index.to_list():  # Convertimos el índice a lista para evitar errores
        raise ValueError(f"Error: Redshift value for '{sn_name}' not found. Check available SN names or old names.")

    redshift = df.loc[sn_name, 'redshift']

    logging.info(f"------Redshift: {redshift}")
    return redshift