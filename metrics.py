import numpy as np


# ==========================================================
# Mean Absolute Error
# ==========================================================
def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


# ==========================================================
# Root Mean Square Error
# ==========================================================
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ==========================================================
# Mean Square Error
# ==========================================================
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


# ==========================================================
# Signal Energy
# ==========================================================
def energy(signal):
    signal = np.asarray(signal)
    return np.sum(signal ** 2)


# ==========================================================
# Signal Power
# ==========================================================
def power(signal):
    signal = np.asarray(signal)
    return np.mean(signal ** 2)


# ==========================================================
# Signal Variance
# ==========================================================
def variance(signal):
    signal = np.asarray(signal)
    return np.var(signal)


# ==========================================================
# Signal Standard Deviation
# ==========================================================
def std(signal):
    signal = np.asarray(signal)
    return np.std(signal)


# ==========================================================
# Peak Value
# ==========================================================
def peak(signal):
    signal = np.asarray(signal)
    return np.max(np.abs(signal))

# ==========================================================
# Signal to Noise Ratio
# ==========================================================


def snr(signal, noise):
    signal_power = power(signal)
    noise_power = power(noise)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)


# ==========================================================
# Packet Interval
# ==========================================================
def packet_interval(time):
    return np.diff(time)


# ==========================================================
# Average Sampling Frequency
# ==========================================================
def sampling_frequency(time):
    dt = packet_interval(time)
    return 1 / np.mean(dt)


# ==========================================================
# Packet Jitter
# ==========================================================
def packet_jitter(time):
    dt = packet_interval(time)
    return np.std(dt)


# ==========================================================
# Packet Stability Index
# ==========================================================
def psi(time):
    dt = packet_interval(time)
    cv = np.std(dt) / np.mean(dt)
    return max(0, 1 - cv)


# ==========================================================
# Motion Enhancement Ratio
# ==========================================================
def mer(static_signal, motion_signal):
    Es = energy(static_signal)
    Em = energy(motion_signal)
    if Es == 0:
        return np.inf
    return Em / Es
