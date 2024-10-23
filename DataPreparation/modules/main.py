import sys
from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import random
import logging
import add_spikes
import shift_spectra

def modify(spectrum):
   modified_spectra = spectrum.copy()
   modifications = random.randint(0,3)
   if modifications == 0:
      logging.info('------Modifications: None')
   elif modifications == 1:
      logging.info('------Modifications: Shift spectra on x axis only')
   elif modifications == 2:
      logging.info('------Modifications: Add spikes only')
   elif modifications == 3:
      logging.info('------Modifications: Shift spectra on x axis AND add spikes')

   return modified_spectra

# --Ensuring reproducibility by setting the seed
random.seed(3312)
