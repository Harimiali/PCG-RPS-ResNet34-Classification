"""
preprocessing.py

Preprocessing utilities for PCG signal classification.

Includes:
- Band-pass filtering
- Spectral subtraction
- Complete preprocessing pipeline

"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: int = 2000,
    lowcut: float = 15.0,
    highcut: float = 800.0,
    order: int = 3
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to a PCG signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D PCG signal.
    sampling_rate : int
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Band-pass filtered signal.
    """

    if signal is None or len(signal) == 0:
        raise ValueError("Input signal is empty.")

    signal = np.asarray(signal, dtype=np.float32)

    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0 or high >= 1 or low >= high:
        raise ValueError("Invalid cutoff frequencies.")

    b, a = butter(order, [low, high], btype="bandpass")
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal.astype(np.float32)


def spectral_subtraction(
    signal: np.ndarray,
    sampling_rate: int = 2000,
    noise_reduction_factor: float = 0.5,
    frame_size: int = 512,
    hop_size: int = 256
) -> np.ndarray:
    """
    Apply spectral subtraction for noise reduction.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    sampling_rate : int
        Sampling frequency in Hz.
    noise_reduction_factor : float
        Weight applied to estimated noise spectrum.
    frame_size : int
        Number of samples per frame.
    hop_size : int
        Step size between frames.

    Returns
    -------
    np.ndarray
        Noise-reduced signal.
    """

    if signal is None or len(signal) == 0:
        raise ValueError("Input signal is empty.")

    signal = np.asarray(signal, dtype=np.float32)

    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive.")

    if len(signal) < frame_size:
        return signal.copy()

    window = np.hanning(frame_size)

    num_frames = 1 + int((len(signal) - frame_size) / hop_size)

    frames = np.zeros((num_frames, frame_size), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frames[i] = signal[start:start + frame_size] * window

    spectrum = np.fft.rfft(frames, axis=1)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Estimate noise from the lowest-energy frames
    frame_energy = np.sum(magnitude ** 2, axis=1)
    noise_frame_count = max(1, int(0.1 * num_frames))
    noise_indices = np.argsort(frame_energy)[:noise_frame_count]
    noise_spectrum = np.mean(magnitude[noise_indices], axis=0)

    cleaned_magnitude = magnitude - noise_reduction_factor * noise_spectrum
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

    cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
    cleaned_frames = np.fft.irfft(cleaned_spectrum, axis=1)

    output = np.zeros(len(signal), dtype=np.float32)
    normalization = np.zeros(len(signal), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        output[start:start + frame_size] += cleaned_frames[i] * window
        normalization[start:start + frame_size] += window ** 2

    nonzero = normalization > 1e-8
    output[nonzero] /= normalization[nonzero]

    return output.astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal amplitude to the range [-1, 1].

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    np.ndarray
        Normalized signal.
    """

    signal = np.asarray(signal, dtype=np.float32)

    max_value = np.max(np.abs(signal))

    if max_value < 1e-8:
        return signal

    return (signal / max_value).astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    sampling_rate: int = 2000,
    lowcut: float = 15.0,
    highcut: float = 800.0,
    filter_order: int = 3,
    apply_spectral_subtraction: bool = True,
    noise_reduction_factor: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for PCG signals.

    Steps:
    1. Convert signal to float32
    2. Remove DC component
    3. Apply Butterworth band-pass filter
    4. Apply spectral subtraction
    5. Normalize amplitude

    Parameters
    ----------
    signal : np.ndarray
        Raw PCG signal.
    sampling_rate : int
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    filter_order : int
        Butterworth filter order.
    apply_spectral_subtraction : bool
        Whether to apply spectral subtraction.
    noise_reduction_factor : float
        Noise subtraction weight.
    normalize : bool
        Whether to normalize final signal.

    Returns
    -------
    np.ndarray
        Preprocessed PCG signal.
    """

    if signal is None or len(signal) == 0:
        raise ValueError("Input signal is empty.")

    processed_signal = np.asarray(signal, dtype=np.float32)

    # Remove DC offset
    processed_signal = processed_signal - np.mean(processed_signal)

    # Band-pass filtering
    processed_signal = bandpass_filter(
        processed_signal,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=filter_order
    )

    # Spectral subtraction
    if apply_spectral_subtraction:
        processed_signal = spectral_subtraction(
            processed_signal,
            sampling_rate=sampling_rate,
            noise_reduction_factor=noise_reduction_factor
        )

    # Final normalization
    if normalize:
        processed_signal = normalize_signal(processed_signal)

    return processed_signal.astype(np.float32)


if __name__ == "__main__":
    # Simple test example
    fs = 2000
    duration = 3
    t = np.linspace(0, duration, fs * duration)

    # Synthetic PCG-like signal + noise
    test_signal = (
        0.6 * np.sin(2 * np.pi * 80 * t)
        + 0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.05 * np.random.randn(len(t))
    )

    processed = preprocess_signal(test_signal, sampling_rate=fs)

    print("Original shape:", test_signal.shape)
    print("Processed shape:", processed.shape)
    print("Processed min:", processed.min())
    print("Processed max:", processed.max())
