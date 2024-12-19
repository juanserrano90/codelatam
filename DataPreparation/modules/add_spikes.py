# -- Authors: Paula Gálvez Molina
# -- Modified from original code by : Willow Fox Fortino
# Last Modified: Oct. 21th, 2024

import numpy as np
from scipy import stats
import logging

# -- Authors: Paula Gálvez Molina
# -- Modified from original code by : Willow Fox Fortino
# Last Modified: Oct. 21th, 2024

def add_spikes(spectrum, redshift, start=2501.69, end=9993.24):
    """Loosely simulate telluric lines by adding in one-pixel wide spikes to the
       spectra are expected de-redshifted telluric lines.

        Args:
        spectrum (float): Supernova spectrum to be modified.
        redshift (float): Redshift of the supernova of the corresponding spectrum.
        start (float): Minimum wavelength of the telluric line.
        end (float): Maximum wavelength of the telluric line.

        Returns:
        spiked (array): Spectrum array with spike(s) added."""

    # First decide how many spikes should be added. At most 5 and at minimum zero.
    num_spikes = stats.randint.rvs(low=0, high=4, size=1).item()

    # Construct the array of spikes that is the same shape as the original spectrum.
    spiked = spectrum.copy() #FBB move this to before tell_loc

    # FBB put this in a if statements
    if num_spikes>0:
      # Next decide the location of the spikes by approximating the first spike at around 760 nm (O2 A-Band) with corresponding redshift.
      # If more than one spike, then the location is given by choosing a random number between 0 and the length of the array containing the spectrum. These other lines intend to simulate artifact.
      tell_loc = 7600 / (redshift + 1)

      # Find the edges of the mask and redefine start and end as those values
      filter = spiked != 0
      nonzero_indices = np.nonzero(filter)[0]
      startSpec, endSpec = nonzero_indices[0], nonzero_indices[-1]

      # The spectrums are in logarithmic scale, so the index of the location of the spike must be determined by the location of the in the logarithmic array closest in value to the tell_loc.
      log_array = np.logspace(np.log10(start), np.log10(end), num=len(spectrum))    # Create the logarithmically spaced array
      # FBB you should not need to recreate this: it should be an object defined in the speclc where you et the spectra from

      idx = (np.abs(log_array - tell_loc)).argmin()     # Find the index of the element closest to the given value
      spike_loc = stats.randint.rvs(low=startSpec, high=endSpec, size=num_spikes) # FBB removed random state: that is set outside of this function
      if np.random.rand() > 0.75: # 1/4 of the times
        spike_loc[0] = idx + np.random.randint(-2,2) # Not necessarily on the pixel of the telluric line but +- 2 pixels

      # Next, we decide if the spike will be an addition or subtraction (i.e., if the telluric line is emission or absorption). 80% of the time, the spike will be in emission (a positive spike) and the rest of the time it will be in absorption.
      spike_dir = stats.binom.rvs(n=1, p=0.80, size=num_spikes)
      spike_dir[spike_dir == 0] = -1

      # Finally choose the magnitude of the spike. We take the absolute value since we are having the sign of the spike determined by the previous set of code.
      # The magnitude is drawn from a gaussian distribution with mean 1 and standard deviation 0.75 so that 68% values lie (0.75, 1.25) as determined by this study.
      spike_mag = np.abs(stats.norm.rvs(loc=1, scale=0.75, size=num_spikes))

      spiked[spike_loc] = spike_mag * spike_dir

      # Logging information
      logging.info(f'------Number of spikes to add: {num_spikes}')
      logging.info(f'------Location of spikes: {spike_loc}')
      logging.info(f'------Magnitude of spikes: {spike_mag}')
      logging.info(f'------Direction of spikes: {spike_dir}')

    return spiked, spike_loc
