from typing import Optional, Tuple, Union

import numpy as np
from scipy import fft, signal
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert


def cross_correlation(x1: np.ndarray, x2: np.ndarray, fs: Optional[int] = None,
                      hilbert_flag: bool = False, normalize_flag: bool = False,
                      freq_range: Optional[Union[np.ndarray, list]] = None,
                      calc_offset: bool = False, plot_flag: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, Union[float, None]]:
    """
    Calculate cross correlation between 2 signals using FFT.
    Output lags are defined relatively to x1, i.e. - if the lag is T, it
    corresponds to inner product between x1(t) and x2(t-T)
    :param x1: the first signal
    :param x2: the second signal
    :param fs: optional, sampling rate. If given - then output lags are given in
    seconds
    :param plot_flag: whether to plot the cross correlation output
    :return: (cc, lags) - where cc contains the cross correlation values and
    lags are the shifts (if fs is given then lags is given in seconds, otherwise
    in samples)
    """

    # preliminaries
    assert (x1.ndim == 1) & (x2.ndim == 1)
    len1 = x1.size
    len2 = x2.size
    len_cc = len1 + len2 - 1

    # compute
    x1_fft = fft.rfft(x1, len_cc)
    x2_fft = fft.rfft(x2, len_cc)
    cc_fft = x1_fft * np.conj(x2_fft)
    if normalize_flag:
        magnitude_factor = np.abs(x1_fft) ** 2
        # magnitude_factor = np.abs(x2_fft) ** 2  # bad, since it is the shorter signal
        # magnitude_factor = np.abs(x1_fft) * np.abs(x2_fft)
        # magnitude_factor += np.percentile(magnitude_factor, 80)
        shaping_factor = 1

        # noise_var_est = np.percentile(np.abs(x2_fft) ** 2, 60)
        # source_var_est = np.maximum(0, np.abs(x2_fft) ** 2 - noise_var_est)
        # shaping_factor = (source_var_est / (noise_var_est + source_var_est))
        k = 2000
        # x2_spec = signal.convolve(np.abs(x2_fft) ** 2, np.ones(k)/k, mode="same")
        x1_spec = signal.convolve(np.abs(x1_fft) ** 2, np.ones(k)/k, mode="same")
        # x1x2_spec = np.abs(signal.convolve(x2_fft.conj()*x1_fft, np.ones(k)/k, mode="same"))
        # mask = (x1_spec > np.percentile(x1_spec, 50)).astype(int)
        # s_spec = (x1x2_spec / np.sqrt(x1_spec)) ** 2
        # n_spec = np.maximum(0, x2_spec - s_spec)
        # # shaping_factor = (s_spec / (s_spec + n_spec)) * mask
        # gamma = x1x2_spec / np.sqrt(x1_spec*x2_spec)
        # shaping_factor = gamma**2 / (1 - gamma ** 2)
        shaping_factor = np.zeros_like(cc_fft)
        shaping_factor[:shaping_factor.size//2] = 1

        cc_fft = (cc_fft / magnitude_factor) * shaping_factor

    if freq_range is not None:
        f_vec = fft.rfftfreq(len_cc, 1/fs)
        assert np.size(freq_range) == 2
        idx = (f_vec < freq_range[0]) | (f_vec > freq_range[1])
        cc_fft[idx] = 0
    cc = fft.irfft(cc_fft, len_cc)
    cc = np.roll(cc, len2 - 1)
    lags = np.arange(len_cc) - (len2 - 1)
    if fs is not None:
        lags = lags / fs
    if hilbert_flag:
        cc = np.abs(hilbert(cc))

    # calculate offset
    if calc_offset:
        offset = lags[np.argmax(np.abs(cc))]
    else:
        offset = None

    # plot
    if plot_flag:
        fig = px.line(y=cc, x=lags)
        lags_units = "samples" if fs is None else "seconds"
        fig.update_xaxes(title=f"Shift [{lags_units}]")
        fig.update_yaxes(title="Cross Correlation")
        fig.show()

    return cc, lags, offset


def fft_plot(x: np.ndarray, fs: Optional[int] = None,
             nfft: int = 2**18, onesided_flag: bool = None,
             mode: str = "magnitude", log_freq_flag: bool = False) -> go.Figure:
    """
    plot fft of a signal

    :param x: signal to analyse
    :param fs: sampling frequency
    :param nfft: number of frequency samples
    :param onesided_flag: if true, only positive frequencies are calculated
    :param mode: either "magnitude", "phase" or "magnitude_phase"
    :param log_freq_flag: flag for plotting frequency axis in log scale,
    can be True only if onesided_flag is True
    :return: plotly figure object
    """

    # input validation
    assert x.ndim == 1, "input must be a 1D array"
    assert mode in ["magnitude", "phase", "magnitude_phase"], \
        "invalid mode, must be magnitude / phase / magnitude_phase"
    if fs is None:
        fs = 2 * np.pi
    if onesided_flag is None:
        if all(np.isreal(x)):
            onesided_flag = True
        else:
            onesided_flag = False
    if log_freq_flag is True:
        assert onesided_flag is True, \
            "log scale can be plotted only if onesided_flag is True"

    # calculate
    nfft = fft.next_fast_len(np.maximum(x.size, nfft))

    if onesided_flag:
        x_fft = fft.rfft(x, n=nfft)
        f_vec = fft.rfftfreq(nfft, 1/fs)
    else:
        x_fft = np.fft.fftshift(fft.fft(x, n=nfft))
        f_vec = np.fft.fftshift(fft.fftfreq(nfft, 1/fs))

    mag = 10*np.log10(np.abs(x_fft)**2)
    phase = np.angle(x_fft) * 180 / (np.pi)

    # plot
    freq_title = "Frequency [rad]" if fs == 2*np.pi else "Frequency [Hz]"

    if mode == "magnitude":
        fig = px.line(x=f_vec, y=mag, log_x=log_freq_flag)
        fig.update_xaxes(title_text=freq_title)
        fig.update_yaxes(title_text="Magnitude [dB]")
    elif mode == "phase":
        fig = px.line(x=f_vec, y=phase, log_x=log_freq_flag)
        fig.update_xaxes(title_text=freq_title)
        fig.update_yaxes(title_text="Phase [degrees]")
    elif mode == "magnitude_phase":
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True)
        fig.add_trace(go.Scatter(x=f_vec, y=mag), row=1, col=1)
        fig.add_trace(go.Scatter(x=f_vec, y=phase), row=2, col=1)
        fig.update_xaxes(title_text=freq_title)
        if log_freq_flag:
            fig.update_xaxes(type="log")
        fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig.update_yaxes(title_text="Phase [degrees]", row=2, col=1)
        fig.update_layout(showlegend=False)

    fig.show()

    return fig


def stft_plot(x: np.ndarray, *args, **kwargs) -> go.Figure:
    """
    Plot stft of a signal
    This function uses the same API of scipy.stft, and plots its output using
    plotly. Note that by default, the output of scipy.stft is normalized
    according to the used window, but this plotting function cancels this
    normalization
    :param x: the signal to plot (numpy array)
    :param args: args for scipy.stft
    :param kwargs: kwargs for scipy.stft
    :return: plotly figure object
    """

    # calculate stft
    f, t, x_stft = signal.stft(x, *args, **kwargs)

    # compensate for scaling factor used in scipy's stft
    if "window" in kwargs.keys():
        if isinstance(kwargs["window"], str):
            window = signal.get_window(kwargs["window"], kwargs["nperseg"])
        else:
            window = kwargs["window"]
        factor = window.sum()
    else:
        factor = kwargs["nperseg"]
    x_stft *= factor

    # plot
    power = 20 * np.log10(np.abs(x_stft))
    fig = px.imshow(power, x=t, y=f, aspect="auto", origin="lower")
    fig.update_yaxes(title="Frequency")
    fig.update_xaxes(title="Time")
    fig.update_layout(coloraxis_colorbar=dict(title="[dB]"))
    fig.show()

    return fig


