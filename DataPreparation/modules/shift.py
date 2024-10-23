def gen_redshift(spectrum):
    """Loosely simulate redshift estimation error by shifting the entire spectrum by at most 5 pixels left or right."""
    shift_amount = stats.randint.rvs(-5, 6, size=1).item()
    shifted_spectrum = np.roll(spectrum, shift_amount)
    logging.info(f'------Shift amount: {shift_amount}')
    return shifted_spectrum