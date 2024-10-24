import sys
from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import random
import logging
from add_spikes import add_spikes
from shift_spectrum import shift_spectrum
from get_redshift import get_redshift

# -- Logging info
logging.basicConfig(filename='modifications.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', filemode='w')

def modify(spectrum, sn_name, delay_phase):
   # --Error handling
   if not isinstance(spectrum, np.ndarray):
      raise TypeError(f"Error: Expected 'np.ndarray', got {type(spectrum).__name__}")
   if not isinstance(sn_name, str):
      raise TypeError(f"Error: Expected 'str', got {type(spectrum).__name__}")
   if not isinstance(delay_phase, float):
      raise TypeError(f"Error: Expected 'float', got {type(spectrum).__name__}")
   
   # --Copy spectrum and log basic information
   modified_spectrum = spectrum.copy()
   logging.info(f'Object: {sn_name}, phase delay: {delay_phase}')

   # --Apply modifications
   modifications = random.randint(0,3)
   if modifications == 0:
      logging.info('---Modifications: None')
   elif modifications == 1:
      logging.info('---Modifications: Shift spectra on x axis only')
      modified_spectrum = shift_spectrum(spectrum)
   elif modifications == 2:
      logging.info('---Modifications: Add spikes only')
      redshift = get_redshift(sn_name)
      modified_spectrum = add_spikes(spectrum, redshift)
   elif modifications == 3:
      logging.info('---Modifications: Shift spectra on x axis AND add spikes')
      redshift = get_redshift(sn_name)
      modified_spectrum = add_spikes(shift_spectrum(spectrum), redshift)

   return modified_spectrum

# --Test
np.random.seed(3312)
url = "https://github.com/juanserrano90/codelatam/raw/main/Data/data/sn_data.parquet"
df_raw = pd.read_parquet(url)
wavelength = np.array([float(c) for c in df_raw.columns[5:]])
column_names = df_raw.columns.values.tolist()
column_names = np.array(column_names[5:]).astype(float)
sn_name = "sn2007uy"
delay_phases = [12.82, 45.82, 54.82, 75.82, 141.82]
sample = []
for delay in delay_phases:
  sample.append(df_raw.loc[(df_raw.index == sn_name) * (df_raw["Spectral Phase"] == delay)].iloc[:, 5:].values[0])
modified = []
for i,s in enumerate(sample):
#   logging.info(f'Object: {sn_name}, phase delay: {delay_phases[i]}')
  modified.append(modify(s, sn_name, delay_phases[i]))
  logging.info(f'\n')

fig, ax = plt.subplots(2, 5, figsize=(20, 10))

# Row 0: Raw spectra
for i in range(5):
    ax[0, i].plot(wavelength, sample[i])

# Row 1: Spikes added spectra
for i in range(5):
    ax[1, i].plot(wavelength, modified[i])

# Add titles to the top of each row
fig.suptitle(f'Spectra comparison of {sn_name}', fontsize=16)
ax[0, 2].set_title('Original Spectra', fontsize=14, loc='center', pad=20)
ax[1, 2].set_title('Modified Spectra', fontsize=14, loc='center', pad=20)

# Add text underneath each plot in row 1
for i in range(5):
    ax[1, i].text(0.5, -0.2, f'Phase delay: {delay_phases[i]}', ha='center', va='center', transform=ax[1, i].transAxes, fontsize=12)

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()