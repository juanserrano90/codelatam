import numpy as np
from scipy import stats
import logging

def shift_spectrum(spectrum, base_redshift):
    """
    Desplaza el espectro según un redshift modificado por un porcentaje aleatorio.
    
    Parameters:
    - spectrum (array): El espectro de la supernova.
    - base_redshift (float): Redshift de referencia de la supernova.
    
    Returns:
    - array: Espectro desplazado.
    """
    # Genera un número aleatorio entre 0 y 1 para el factor de escala
    random_factor = np.random.uniform(0, 1)
    
    # Calcula el redshift modificado como un porcentaje del base_redshift
    shift_redshift = base_redshift * random_factor
    
    # Factor de desplazamiento para las longitudes de onda
    scaling_factor = 1 + shift_redshift - base_redshift
    
    # Desplazamos el espectro usando el redshift modificado (factor de longitud de onda)
    wavelengths = np.arange(len(spectrum))
    shifted_wavelengths = wavelengths * scaling_factor
    
    # Interpolamos el espectro para el nuevo rango de longitudes de onda
    shifted_spectrum = np.interp(shifted_wavelengths, wavelengths, spectrum, left=0, right=0)
    
    # Información de depuración
    logging.info(f'------Redshift modificado (factor aleatorio): {shift_redshift}')
    
    return shifted_spectrum
