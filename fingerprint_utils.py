from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from tqdm import trange


def calc_spectrum(y: np.ndarray, **stft_args) -> pd.DataFrame:
    """
    calculate spectrum from a given signal
    :param y: single-channel signal
    :param stft_args: arguments for scipy.signal.stft function
    :return: spectrum (in dB) as a dataframe, with time and frequency values
    (rows -> freq, cols -> time)
    """

    # calc spectrum
    f_vec, t_vec, stft = signal.stft(y, **stft_args)

    # round freq and time values
    f_vec = np.round(f_vec, 0)
    t_vec = np.round(t_vec, 6)

    # convert stft to dB values
    stft = np.abs(stft)
    stft[stft == 0] = 1e-20
    stft = 20*np.log10(stft)

    # save as dataframe (for named rows/columns)
    spec = pd.DataFrame(stft, index=f_vec, columns=t_vec)

    return spec


def get_spec_peaks(spec: Union[np.ndarray, pd.DataFrame],
                   connectivity_mask: int, peak_neighborhood_size: int,
                   min_val_relative: float = -np.inf,
                   min_freq: float = -np.inf, max_freq: float = -np.inf,
                   plot_flag: bool = False, verbose: bool = False
                   ) -> pd.DataFrame:

    """
    extract maximum peaks from a spectrum
    :param spec: the signal spectrum (in dB), shape [freq, time]
    :param connectivity_mask: 1 or 2, way of defining neighbors for each point
    in the spectrum (see scipy.ndimage.morphology.generate_binary_structure).
    Choosing 2 is recommended as it is faster
    :param peak_neighborhood_size: for dilation using
    scipy.ndimage.iterate_structure
    :param min_val_relative: a minimal value for a valid peak, defined
    relatively to (approx) the maximum of the spectrum. For example - if
    min_val_relative is 60, and max(spec) = -30, then only values above -90
    are allowed
    :param min_freq: minimal frequency for a valid peak
    :param max_freq: maximal frequency for a valid peak
    :param plot_flag: boolean flag, whether to plot results
    :param verbose: boolean flag, whether to print messages
    :return: a dataframe containing all peaks, with columns - freq, freq_idx,
     time and time_idx
    """

    # preliminaries
    if not isinstance(spec, pd.DataFrame):
        spec = pd.DataFrame(spec)
    arr2d = spec.values

    # get connectivity
    struct = generate_binary_structure(2, connectivity_mask)

    # apply dilation
    neighborhood = iterate_structure(struct, peak_neighborhood_size)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2d, footprint=neighborhood) == arr2d

    # applying erosion
    background = (arr2d == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # boolean mask of arr2d with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    vals = arr2d[detected_peaks]
    freq_peaks_idx, time_peaks_idx = np.where(detected_peaks)

    # filter invalid peaks
    vals = vals.flatten()
    f_vec = spec.index.values
    t_vec = spec.columns.values
    largest_val_approx = np.percentile(arr2d, 99.9)
    # use percentile instead of max to avoid outliers
    min_val = largest_val_approx + min_val_relative

    mask = (vals > min_val) & (f_vec[freq_peaks_idx] >= min_freq) & (
            f_vec[freq_peaks_idx] <= max_freq)
    mask = np.where(mask)[0]

    freq_peaks_idx = freq_peaks_idx[mask]
    time_peaks_idx = time_peaks_idx[mask]

    # convert to freq and time values and wrap as dataframe
    freq_peaks = f_vec[freq_peaks_idx]
    time_peaks = t_vec[time_peaks_idx]
    peaks = pd.DataFrame({"freq": freq_peaks, "freq_idx": freq_peaks_idx,
                          "time": time_peaks, "time_idx": time_peaks_idx})

    # print and plot
    if verbose:
        print(f"Total number of located peaks: {len(mask)}")
        peaks_per_sec = len(mask) / t_vec[-1]
        print(f"Average number of peaks per second: {peaks_per_sec}")

    if plot_flag:
        plot_spec_peaks(spec, peaks)

    return peaks


def generate_patterns(peaks: pd.DataFrame, fan_value: int,
                      min_time_delta: float, max_time_delta: float,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Generate patterns from a given set of peaks in the spectrum, in order to use
    them later as fingerprints of the signal. Each pattern is defined by
    2 peaks: frequency of peak1, frequency of peak2 and the time difference
    between them. Therefore, the pattern is shift-invariant.

    :param peaks: a dataframe of the peaks in the spectrum, each row is a peak
    and its location is described by "time" and "freq" columns
    :param fan_value: how many peaks will be examined as a potential pair for
    each peak
    :param min_time_delta: minimum time difference allowed for a pair of peaks
    :param max_time_delta: maximum time difference allowed for a pair of peaks
    :param verbose: boolean flag, whether to print messages or not
    :return: a dataframe with all the generated patterns, and their absolute
    time in the signal (columns: freq1, freq2, t_delta and t_abs)
    """

    # preliminaries
    patterns = {
        "freq1": [],
        "freq2": [],
        "t_delta": [],
        "t_abs": []
    }

    if verbose:
        print("Generating patterns..")

    # sort peaks
    peaks = peaks.sort_values(by=["time", "freq"])
    num_peaks = peaks.shape[0]

    # generate pairs
    # TODO: can be optimized and a smart algorithm can be chosen,
    #  this is inefficient and may be sensitive to noise
    for i in trange(num_peaks):
        for j in range(1, fan_value):
            if (i + j) < num_peaks:

                freq1 = peaks.freq.iat[i]
                freq2 = peaks.freq.iat[i + j]
                t1 = peaks.time.iat[i]
                t2 = peaks.time.iat[i + j]
                t_delta = t2 - t1

                if min_time_delta <= t_delta <= max_time_delta:
                    patterns["freq1"].append(freq1)
                    patterns["freq2"].append(freq2)
                    patterns["t_delta"].append(t_delta)
                    patterns["t_abs"].append(t1)

    patterns_df = pd.DataFrame.from_dict(patterns)

    if verbose:
        print(f"Generated {patterns_df.shape[0]} patterns in total")

    return patterns_df


def plot_spec_peaks(spec: pd.DataFrame, peaks: pd.DataFrame):
    """
    plot spectrum peaks
    :param spec: spectrum (in dB), given as a dataframe with time and frequency
    values (rows -> freq, cols -> time)
    :param peaks: a dataframe of the peaks in the spectrum, where each row is
    a peak, and its location is described by "time" and "freq" columns
    """

    # prepare values
    arr2d = spec.values
    f_vec = spec.index.values
    t_vec = spec.columns.values

    # prepare range of colorbar
    range_max = np.percentile(arr2d, 99.9)
    range_min = range_max - 70  # 70 dB range for visualization

    # plot
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=arr2d, x=t_vec, y=f_vec, zmin=range_min,
                             zmax=range_max, colorbar={"title": "[dB]"}))
    fig.add_trace(go.Scatter(x=peaks["time"], y=peaks["freq"],
                             mode="markers", marker_color="black"))
    dt = t_vec[1] - t_vec[0]
    df = f_vec[1] - f_vec[0]
    fig.update_xaxes(range=[t_vec[0] - dt / 2, t_vec[-1] + dt / 2],
                     title="Time [sec]")
    fig.update_yaxes(range=[f_vec[0] - df / 2, f_vec[-1] + df / 2],
                     title="Frequency [Hz]")
    fig.show()
