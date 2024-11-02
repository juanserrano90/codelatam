import numpy as np
from scipy import stats
import logging

def shift_spectrum(spectrum, base_redshift, max_error_percentage=1):
    """
    Desplaza el espectro según un redshift modificado por un porcentaje aleatorio.
    
    Parameters:
    - spectrum (array): El espectro de la supernova.
    - base_redshift (float): Redshift de referencia de la supernova.
    - max_error_percentage (float): Máximo porcentaje de error en el redshift (default: 0.1).
    
    Returns:
    - array: Espectro desplazado.
    """
    # Genera un número aleatorio entre 0 y 1
    random_factor = np.random.uniform(0, 1)
    # Calcula el redshift modificado con el factor aleatorio
    shift_redshift = base_redshift * (1 + random_factor * max_error_percentage)
    
    # Desplazamos el espectro usando el redshift modificado
    shifted_spectrum = np.interp(
        np.arange(len(spectrum)) * (1 + shift_redshift - base_redshift),
        np.arange(len(spectrum)),
        spectrum,
        left=0,
        right=0
    )
    
    logging.info(f'------Redshift modificando (factor aleatorio): {shift_redshift}')
    return shifted_spectrum