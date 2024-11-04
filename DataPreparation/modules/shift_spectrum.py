import numpy as np
from scipy import stats
import logging

def shift_spectrum(spectrum, base_redshift, max_error_percentage=1):
    """
    Desplaza el espectro según un redshift modificado por un porcentaje aleatorio.
    
    Parameters:
    - spectrum (array): El espectro de la supernova.
    - base_redshift (float): Redshift de referencia de la supernova.
    - max_error_percentage (float): Máximo porcentaje de error en el redshift (default: 1).
    
    Returns:
    - array: Espectro desplazado.
    """
    # Verificación de entrada
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("El espectro debe ser un numpy array.")
    if len(spectrum) == 0:
        raise ValueError("El espectro no puede estar vacío.")
    
    # Genera un número aleatorio entre 0 y 1
    random_factor = np.random.uniform(0, 1)
    # Calcula el redshift modificado con el factor aleatorio
    shift_redshift = int(base_redshift * random_factor * len(spectrum))  # Convierte a entero para usarlo con roll
    
    # Desplazamos el espectro usando el redshift modificado
    shifted_spectrum = np.roll(spectrum, shift_redshift)

    logging.info(f'------Redshift modificando (factor aleatorio): {shift_redshift}')
    return shifted_spectrum
