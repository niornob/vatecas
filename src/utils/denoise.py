import pandas as pd
import numpy as np
import pywt


def wavelet_denoise(series, wavelet='db4', level=None, threshold_method='soft') -> pd.Series:
    """
    Denoise a pandas Series using discrete wavelet transform.
    
    Parameters:
    - series: pandas.Series
    - wavelet: str, wavelet name (e.g., 'db4')
    - level: int or None, decomposition level
    - threshold_method: 'soft' or 'hard'
    
    Returns:
    - pandas.Series of the reconstructed (denoised) signal
    """
    coeffs = pywt.wavedec(series, wavelet, mode='symmetric', level=level)
    # Estimate a universal threshold based on the detail coefficients at the first level
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-1]))
    uthresh = sigma * np.sqrt(2 * np.log(len(series)))
    
    # Threshold detail coefficients
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=threshold_method) for c in coeffs[1:]
    ]
    # Reconstruct the signal
    reconstructed = pywt.waverec(denoised_coeffs, wavelet, mode='symmetric')
    return pd.Series(list(reconstructed)[:len(series)], index=series.index)



if __name__ == "__main__":
    import matplotlib
    matplotlib.use('tkagg') 
    import matplotlib.pyplot as plt

    x = np.array(np.linspace(0, 1, 500))
    signal = np.sin(4 * np.pi * x) + 0.5 * np.random.randn(500)
    series = pd.Series(signal)

    denoised = wavelet_denoise(series, wavelet='db4', level=4, threshold_method='soft')

    # Plot original and denoised series
    plt.figure(figsize=(12,6))
    plt.plot(series, label='Original')
    plt.plot(denoised, label='Denoised')
    plt.legend()
    plt.title('Wavelet Denoising of a Pandas Series')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()