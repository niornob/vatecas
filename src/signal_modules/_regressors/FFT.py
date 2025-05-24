import numpy as np
from scipy.signal import find_peaks
from typing import Optional, Tuple, Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from signal_modules._regressors._oracle_base import Oracle


class FFTFilter(Oracle):
    """
    Enhanced FFT-based time series prediction oracle with multi-series support.
    
    This class uses Fast Fourier Transform to decompose time series into frequency
    components, identifies significant periodic patterns, and reconstructs predictions
    by extrapolating these patterns into the future. The approach is particularly
    effective for time series with strong periodic or cyclical behavior.
    """
    
    # Declare instance attributes with types
    perc: float
    short_period_threshold: float
    long_period_threshold: float

    def __init__(
        self,
        name: str = "FFTFilter",
        version: str = "2.0",
        params: Optional[Dict[str, object]] = None,
        perc: float = 0.0,
        short_period_threshold: float = 0.0,
        long_period_threshold: float = float("inf"),
    ) -> None:
        """
        Initialize the FFT Filter oracle.
        
        Args:
            name: Name identifier for this oracle instance
            version: Version string for tracking
            params: Additional parameters dictionary (optional)
            perc: Percentile threshold for amplitude filtering (0-1). Higher values
                 keep only the strongest frequency components
            short_period_threshold: Minimum period length to include in reconstruction.
                                  Filters out very high-frequency noise
            long_period_threshold: Maximum period length to include in reconstruction.
                                 Filters out very low-frequency trends
        """
        super().__init__(name, version, params)
        
        assert 0 <= perc <= 1, "FreqFilter percentile must be between 0 and 1."
        self.perc = perc
        self.short_period_threshold = short_period_threshold
        self.long_period_threshold = long_period_threshold
        self.wavelet = self.params.get('wavelet', 'db4')
        self.level = self.params.get('level', None),
        self.threshold_method = str(self.params.get('threshold_method', "soft"))


    def _predict(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Predict the next time step values for all variables using FFT analysis.
        
        This method applies FFT-based prediction to each time series independently,
        identifying significant frequency components and extrapolating them to
        generate predictions for the next time step.
        
        Args:
            data: Dictionary where keys are variable names and values are pandas Series
                 containing the time history of each variable. All Series should have
                 the same length and represent the same time steps.
        
        Returns:
            np.ndarray: 1D array containing the predicted values for the next time step,
                       with the same order as would be created by pd.DataFrame(data).columns
        
        Raises:
            ValueError: If the input data is empty or contains Series of different lengths
        """
        # Input validation
        if not data:
            raise ValueError("Input data dictionary cannot be empty")
        
        # Check that all Series have the same length
        series_lengths = [len(series) for series in data.values()]
        if len(set(series_lengths)) > 1:
            raise ValueError("All pandas Series in data must have the same length")
        
        if min(series_lengths) == 0:
            raise ValueError("Cannot predict from empty time series")
        
        # Convert to DataFrame to ensure consistent ordering
        df = pd.DataFrame(data)
        predictions = []
        
        # Apply FFT prediction to each series independently
        for column_name in df.columns:
            series = df[column_name]
            
            try:
                # Apply the same FFT prediction logic as the original single-series method
                prediction = self._predict_single_series(series)
                predictions.append(prediction)
                
            except Exception as e:
                # Handle prediction errors gracefully
                print(f"Warning: FFT prediction failed for series '{column_name}': {str(e)}")
                # Use the last available value as fallback
                predictions.append(float(series.iloc[-1]))
        
        return np.array(predictions)
    
    def _predict_single_series(self, series: pd.Series) -> float:
        """
        Apply FFT-based prediction to a single time series.
        
        This method encapsulates the core FFT prediction logic, making it easier
        to apply consistently across multiple series and handle errors gracefully.
        """
        coeffs = pywt.wavedec(
            series, self.wavelet, mode="symmetric", level=None
        )
        sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1]))
        uthresh = sigma * np.sqrt(2 * np.log(len(series)))
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=uthresh, mode=self.threshold_method)
            for c in coeffs[1:]
        ]
        smooth_series = pd.Series(pywt.waverec(denoised_coeffs, self.wavelet, mode="symmetric")).iloc[:len(series)]
        
        # Compute FFT spectrum and identify significant peaks
        _, _, fft_vals, peaks_idx, peaks_freq, _ = self._compute_fft_spectrum_peaks(
            np.asarray(smooth_series.values)
        )
        
        # Apply period-based filtering to focus on relevant frequency components
        idx_masked = peaks_idx[
            np.logical_and(
                1 / peaks_freq > self.short_period_threshold,
                1 / peaks_freq < self.long_period_threshold,
            )
        ]
        
        # Reconstruct time series with one additional point (the prediction)
        reconstructed = self._reconstruct_time_series_from_peaks(
            fft_vals, idx_masked, len(smooth_series) + 1
        )
        
        # Return the final point as our prediction
        return float(reconstructed[-1])

    def _compute_fft_spectrum_peaks(
        self, time_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FFT spectrum and identify significant peaks.
        
        This method transforms the time series into the frequency domain and identifies
        the most significant frequency components based on amplitude thresholds.
        
        Returns:
            Tuple containing: frequencies, amplitudes, complex FFT values, 
            peak indices, peak frequencies, and peak amplitudes
        """
        x = np.asarray(time_series, dtype=float)

        N = len(x)
        fft_vals = np.fft.rfft(x)  # Use real FFT since input is real-valued
        amp = np.abs(fft_vals)     # Compute amplitude spectrum
        freqs = np.fft.rfftfreq(N, d=1.0)  # Frequency bins

        # Set up peak detection parameters
        pk_kwargs: Dict[str, Any] = {}
        if self.perc is not None and self.perc > 0:
            # Use percentile-based amplitude threshold
            # Higher percentiles mean we keep fewer, but stronger, frequency components
            percentile_threshold = self.perc * 100
            amplitude_threshold = np.percentile(amp, percentile_threshold)
            pk_kwargs["height"] = amplitude_threshold

        # Find significant peaks in the amplitude spectrum
        peaks_idx, _ = find_peaks(amp, **pk_kwargs)
        peaks_freqs = freqs[peaks_idx]
        peaks_amp = amp[peaks_idx]

        return freqs, amp, fft_vals, peaks_idx, peaks_freqs, peaks_amp

    def _reconstruct_time_series_from_peaks(
        self, fft_vals: np.ndarray, peaks_idx: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Reconstruct time series using only selected frequency components.
        
        This method creates a filtered version of the original FFT that includes
        only the significant frequency components, then transforms back to the
        time domain to generate the reconstructed series.
        """
        # Create a filtered FFT that includes only the selected peaks
        filtered_fft = np.zeros_like(fft_vals)
        filtered_fft[peaks_idx] = fft_vals[peaks_idx]
        
        # Transform back to time domain with the desired number of samples
        return np.fft.irfft(filtered_fft, n=n_samples)

    def diagnostics(self, data: Dict[str, pd.Series], window: Optional[int] = None, **kwargs) -> None:
        """
        Perform rolling forecast diagnostics on the FFT filter oracle.
        
        This method provides comprehensive diagnostic analysis similar to the KalmanFilter
        diagnostics, including rolling predictions, visualization, and performance metrics.
        The analysis helps understand how well the FFT-based approach captures and
        predicts the periodic patterns in your time series data.
        
        Args:
            data: Dictionary where keys are variable names and values are pandas Series
                 containing the complete time history of each variable
            window: Size of the rolling window for predictions. If None, uses the
                   long_period_threshold as a reasonable default window size
            **kwargs: Additional plotting or analysis parameters
        """
        # Input validation
        if not data:
            raise ValueError("Input data dictionary cannot be empty")
        
        # Convert to DataFrame for easier handling
        actual_df = pd.DataFrame(data)
        dates = actual_df.index
        variables = actual_df.columns
        
        # Validate that we have enough data points for meaningful analysis
        if len(dates) < 2:
            raise ValueError("Need at least 2 data points for diagnostic analysis")
        
        if window is None:
            window = len(actual_df)
        
        print(f"Running FFT rolling forecast analysis with window size: {window}")
        print(f"FFT parameters: perc={self.perc}, short_period={self.short_period_threshold}, long_period={self.long_period_threshold}")
        print(f"Variables being analyzed: {list(variables)}")
        
        # Initialize predictions DataFrame
        preds_df = pd.DataFrame(self.regress(data=data, window=window))
        
        # Generate diagnostic plots and metrics
        self._generate_diagnostic_plots(actual_df, preds_df, dates, variables)
        self._calculate_performance_metrics(actual_df, preds_df, variables)
        
        # Generate frequency domain diagnostics for additional insight
        self._generate_frequency_diagnostics(actual_df, variables)
    
    def _generate_diagnostic_plots(self, actual_df: pd.DataFrame, preds_df: pd.DataFrame, 
                                 dates: pd.Index, variables: pd.Index) -> None:
        """Generate comparison plots for actual vs predicted values."""

        for variable in variables:
            plt.figure(figsize=(14, 8))
            
            # Create subplot layout for better visualization
            plt.subplot(2, 1, 1)
            
            # Plot actual vs predicted time series
            plt.plot(dates, actual_df[variable], 
                    label=f"Actual {variable}", 
                    linewidth=2, alpha=0.8, color='blue')
            
            plt.plot(dates, preds_df[variable], 
                    label="1-Step FFT Forecast", 
                    linestyle='--', linewidth=1.5, alpha=0.7, color='red')
            
            plt.title(f"FFT-Based One-Step-Ahead Forecast vs Actual — {variable}", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Time", fontsize=12)
            plt.ylabel(f"{variable} Value", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Plot prediction errors in second subplot
            plt.subplot(2, 1, 2)
            errors = actual_df[variable].reset_index(drop=True) - preds_df[variable]
            plt.plot(dates, errors, 
                    label="Prediction Error", 
                    linewidth=1, alpha=0.7, color='green')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f"Prediction Errors — {variable}", fontsize=12)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Error", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _calculate_performance_metrics(self, actual_df: pd.DataFrame, 
                                     preds_df: pd.DataFrame, variables: pd.Index) -> None:
        """Calculate and display performance metrics for the predictions."""
        
        print("\n" + "="*70)
        print("FFT FILTER PREDICTION PERFORMANCE METRICS")
        print("="*70)
        
        for variable in variables:
            # Skip first value since we don't have a real prediction for it
            actual_vals = actual_df[variable].iloc[1:].values
            pred_vals = preds_df[variable].iloc[1:].values
            
            # Calculate performance metrics
            mse = np.mean((actual_vals - pred_vals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_vals - pred_vals))
            
            # Calculate percentage-based metrics (handle division by zero)
            non_zero_actuals = actual_vals[actual_vals != 0]
            non_zero_preds = pred_vals[actual_vals != 0]
            if len(non_zero_actuals) > 0:
                mape = np.mean(np.abs((non_zero_actuals - non_zero_preds) / non_zero_actuals)) * 100
            else:
                mape = float('inf')
            
            # Calculate correlation
            if np.std(actual_vals) > 0 and np.std(pred_vals) > 0:
                correlation = np.corrcoef(actual_vals, pred_vals)[0, 1]
            else:
                correlation = float('nan')
            
            print(f"\nVariable: {variable}")
            print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
            print(f"  Mean Absolute Error (MAE):     {mae:.6f}")
            if np.isfinite(mape):
                print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            else:
                print(f"  Mean Absolute Percentage Error (MAPE): N/A (zero values present)")
            if np.isfinite(correlation):
                print(f"  Correlation with actual values: {correlation:.4f}")
            else:
                print(f"  Correlation with actual values: N/A (constant values)")
        
        print("\n" + "="*70)
    
    def _generate_frequency_diagnostics(self, actual_df: pd.DataFrame, variables: pd.Index) -> None:
        """Generate frequency domain diagnostic plots to understand FFT behavior."""
        
        print("\nGenerating frequency domain diagnostics...")
        
        for variable in variables:
            series = actual_df[variable]
            
            # Compute FFT analysis for the full series
            freqs, amp, fft_vals, peaks_idx, peaks_freq, peaks_amp = self._compute_fft_spectrum_peaks(
                np.asarray(series.values)
            )
            
            # Create frequency domain visualization
            plt.figure(figsize=(14, 10))
            
            # Plot 1: Full amplitude spectrum
            plt.subplot(3, 1, 1)
            plt.plot(freqs, amp, linewidth=1, alpha=0.7, label='Full Spectrum')
            plt.scatter(peaks_freq, peaks_amp, color='red', s=50, alpha=0.8, 
                       label=f'Detected Peaks ({len(peaks_idx)})')
            plt.title(f"FFT Amplitude Spectrum — {variable}", fontsize=14, fontweight='bold')
            plt.xlabel("Frequency")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale often reveals more detail
            
            # Plot 2: Period domain (more intuitive for time series analysis)
            plt.subplot(3, 1, 2)
            periods = 1 / freqs[1:]  # Skip DC component
            period_amp = amp[1:]
            plt.plot(periods, period_amp, linewidth=1, alpha=0.7, label='Period Spectrum')
            
            # Highlight the period range we're using for prediction
            period_peaks = 1 / peaks_freq[peaks_freq > 0]  # Avoid division by zero
            period_peaks_amp = peaks_amp[peaks_freq > 0]
            
            # Apply period filtering to show what we actually use
            period_mask = np.logical_and(
                period_peaks > self.short_period_threshold,
                period_peaks < self.long_period_threshold
            )
            used_periods = period_peaks[period_mask]
            used_period_amps = period_peaks_amp[period_mask]
            
            plt.scatter(used_periods, used_period_amps, color='green', s=50, alpha=0.8,
                       label=f'Used for Prediction ({len(used_periods)})')
            plt.scatter(period_peaks[~period_mask], period_peaks_amp[~period_mask], 
                       color='orange', s=30, alpha=0.6, label='Filtered Out')
            
            plt.title(f"Period Domain Analysis — {variable}")
            plt.xlabel("Period (time units)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            
            # Plot 3: Reconstructed vs original for visual validation
            plt.subplot(3, 1, 3)
            # Use only the filtered peaks for reconstruction
            filtered_peaks_idx = peaks_idx[np.logical_and(
                1 / peaks_freq > self.short_period_threshold,
                1 / peaks_freq < self.long_period_threshold,
            )]
            reconstructed = self._reconstruct_time_series_from_peaks(
                fft_vals, filtered_peaks_idx, len(series)
            )
            
            # Show recent portion for clarity
            recent_start = max(0, len(series) - 100)
            recent_dates = series.index[recent_start:]
            recent_original = series.iloc[recent_start:].values
            recent_reconstructed = reconstructed[recent_start:]
            
            plt.plot(recent_dates, recent_original, 
                    label='Original', linewidth=2, alpha=0.8)
            plt.plot(recent_dates, recent_reconstructed, 
                    label='FFT Reconstruction', linewidth=1.5, alpha=0.7, linestyle='--')
            plt.title(f"Signal Reconstruction Quality (Recent Data) — {variable}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()