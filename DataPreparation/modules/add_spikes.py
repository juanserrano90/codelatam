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
       spectra, expected de-redshifted telluric lines."""

    # Decide how many spikes should be added. At most 5 and at minimum zero.
    num_spikes = stats.randint.rvs(low=0, high=4, size=1).item()

    # Construct the array of spikes that is the same shape as the original spectrum.
    spiked = spectrum.copy()  # Make a copy of the spectrum to modify

    if num_spikes > 0:
        # Approximate the first spike at around 760 nm (O2 A-Band) with corresponding redshift.
        tell_loc = 7600 / (redshift + 1)

        # Find the edges of the mask and redefine start and end as those values
        filter = spiked != 0
        nonzero_indices = np.nonzero(filter)[0]
        startSpec, endSpec = nonzero_indices[0], nonzero_indices[-1]

        # The spectrums are in logarithmic scale, so the index of the location of the spike must be determined by the location of the in the logarithmic array closest in value to the tell_loc.
        log_array = np.logspace(np.log10(startSpec), np.log10(endSpec), num=len(spectrum))
        idx = np.clip((np.abs(log_array - tell_loc)).argmin(), startSpec, endSpec - 1) #ment closest to the given value
        spike_loc = stats.randint.rvs(low=startSpec, high=endSpec, size=num_spikes)  # Random spike locations within spectrum
        spike_loc = np.clip(spike_loc, 0, len(spiked)-1)  # Ensure spike locations are within valid indices

        if np.random.rand() > 0.75:  # 1/4 of the times
            spike_loc[0] = idx + np.random.randint(-2, 2)  # Adjust the first spike location by +- 2 pixels

        # Decide if the spike will be in emission or absorption (80% emission, 20% absorption)
        spike_dir = stats.binom.rvs(n=1, p=0.80, size=num_spikes)
        spike_dir[spike_dir == 0] = -1

        # Magnitude of the spike drawn from a Gaussian distribution
        spike_mag = np.abs(stats.norm.rvs(loc=1, scale=0.75, size=num_spikes))

        # Add the spikes to the spectrum
        spiked[spike_loc] = spike_mag * spike_dir

        # Logging information
        logging.info(f'Number of spikes to add: {num_spikes}')
        logging.info(f'Location of spikes: {spike_loc}')
        logging.info(f'Magnitude of spikes: {spike_mag}')
        logging.info(f'Direction of spikes: {spike_dir}')

    return spiked  # Return the modified spectrum
