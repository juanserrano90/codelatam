import numpy as np
from scipy import stats
import logging

def shift_spectrum(spectrum, base_redshift, max_error_percentage=10, wavelength_range=(2501, 9993.24)):
    """
    Shifts the spectrum according to a randomly modified redshift within a ±10% range of the base redshift.
    
    Parameters:
    - spectrum (array): The supernova spectrum.
    - base_redshift (float): Reference redshift of the supernova.
    - max_error_percentage (float): Maximum error percentage in the redshift (default: 10).
    - wavelength_range (tuple): Estimated wavelength range in Å (min, max).
    
    Returns:
    - array: Shifted spectrum.
    """
    # Input validation
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("The spectrum must be a numpy array.")
    if len(spectrum) == 0:
        raise ValueError("The spectrum cannot be empty.")
    
    # Get the minimum and maximum wavelength from the specified range
    wavelength_min, wavelength_max = wavelength_range
    total_pixels = len(spectrum)
    
    # Calculate the value of a (wavelength increment per pixel) and b (wavelength at the first pixel)
    a = (wavelength_max - wavelength_min) / (total_pixels - 1)
    b = wavelength_min
    
    # Calculate the central wavelength using linear calibration
    central_pixel = total_pixels // 2
    central_wavelength = a * central_pixel + b
    logging.info(f"Calculated central wavelength: {central_wavelength} Å")
    print(f'Central wavelength: {central_wavelength} Å')
    
    # Generate a random redshift variation within ±10%
    redshift_variation = np.random.uniform(-max_error_percentage, max_error_percentage) / 100
    modified_redshift = base_redshift * (1 + redshift_variation)
    print('Redshift variation:', modified_redshift)
    
    # Shift in wavelength
    shift_wavelength = central_wavelength * modified_redshift
    logging.info(f"Wavelength shift: {shift_wavelength} Å")
    print(f'Wavelength shift: {shift_wavelength} Å')
    
    # Calculate the pixel shift
    wavelength_per_pixel = a
    shift_pixels = int(shift_wavelength / wavelength_per_pixel)
    logging.info(f"Pixel shift: {shift_pixels}")
    print(f'Pixel shift: {shift_pixels}')
    
    # Shift the spectrum using the calculated pixel shift
    shifted_spectrum = np.roll(spectrum, shift_pixels)

    logging.info(f'Modified redshift (random variation): {modified_redshift}')
    return shifted_spectrum
