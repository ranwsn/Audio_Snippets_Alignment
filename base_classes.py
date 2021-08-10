import pickle
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
from scipy import signal
from scipy.stats import entropy

from config import ConfigFP
from fingerprint_utils import get_spec_peaks, calc_spectrum, generate_patterns,\
    plot_spec_peaks


class AudioSignal:
    """
    A single-channel audio signal object
    """

    def __init__(self, config: ConfigFP, y: np.ndarray, fs: int,
                 audio_id: Optional[int] = None):
        """
        :param config: configuration
        :param y: signal
        :param fs: sample-rate
        :param audio_id: optional, give id to audio signal
        """

        # validate single-channel
        assert (y.ndim == 1), "signal must be single channel"

        # convert sample rate to desired value
        fs_target = config.fs
        self.y = signal.resample_poly(y, fs_target, fs)

        self.config = config
        self.audio_id = audio_id
        self.fingerprint = None

    def plot(self) -> go.Figure:
        """plot signal in time domain"""

        t_vec = np.arange(self.y.size) / self.config.fs
        fig = px.line(y=self.y, x=t_vec)
        fig.update_xaxes(title="Time [sec]")
        fig.show()

        return fig

    def get_signal(self) -> np.ndarray:
        return self.y

    def get_fingerprint(self):
        assert self.has_fingerprint(), "fingerprint wasn't calculated yet"
        return self.fingerprint

    def has_fingerprint(self) -> bool:
        return self.fingerprint is not None

    def generate_fingerprint(self):
        """calculate fingerprint of the signal"""

        self.fingerprint = Fingerprint(config=self.config, y=self,
                                       audio_id=self.audio_id)

    def save_fingerprint(self, path: Union[str, Path]):
        """save finger print to file"""

        assert self.has_fingerprint(), "fingerprint wasn't calculated yet"
        self.fingerprint.save(path)

    def plot_fingerprint(self):
        """plot fingerprint (spectrum and peaks)"""

        assert self.has_fingerprint(), "fingerprint wasn't calculated yet"

        c = self.config
        spec = calc_spectrum(self.get_signal(), **c.stft_args, fs=c.fs)
        peaks = self.fingerprint.get_peaks()
        plot_spec_peaks(spec, peaks)

    @classmethod
    def from_file(cls, path: Union[str, Path], config: ConfigFP,
                  audio_id: Optional[int] = None):
        """load signal from file"""

        path = Path(path)
        y, fs = sf.read(path)

        return cls(config=config, y=y, fs=fs, audio_id=audio_id)


class FingerprintDatabase:
    """
    A database containing fingerprints of an AudioSignal class.
    Wrapped as a separate class than Fingerprint, in order to implement here
    the indexing and the filtering of the data (performance optimization should
    be performed here, keeping the same API)
    """
    def __init__(self, data: pd.DataFrame):
        """
        :param data: a dataframe with all the fingerprints of a given
        AudioSignal, containing 4 columns - 'freq1', 'freq2', 't_delta', 't_abs'
        """
        self.data = self._convert2db(data)

    def search_fingerprints(self, fp_to_search: 'FingerprintDatabase'
                            ) -> pd.DataFrame:
        """
        search for fingerprints matches to the current fingerprints
        :param fp_to_search: a database of fingerprints to be searched inside
        the current fingerprints
        :return: a dataframe with the fingerprints matching results,
        including all matches and also fingerprints in the current database
        that didn't have any match (left join)
        """
        db_base = self.data
        db_search = fp_to_search.data
        matches = db_base.merge(
            db_search, how="left",
            left_on=['freq1', 'freq2', 't_delta'],
            right_on=['freq1', 'freq2', 't_delta'],
            suffixes=["_base", "_found"]
        )

        return matches

    def _convert2db(self, data: pd.DataFrame):
        """convert data to high performance database format (index for good
        search performance)"""
        # TODO: may be expensive in terms of memory (4 levels of hasing are
        #  not needed). I added t_abs to the index so that the index will be
        #  unique and therefore search will be O(1). This can be optimized,
        #  high search speed can still be gained with less hashing

        db = data.set_index(['freq1', 'freq2', 't_delta', 't_abs'], drop=False
                            ).sort_index()
        db = db["t_abs"].to_frame()

        return db


class Fingerprint:
    """A fingerprint of an AudioSignal class"""
    def __init__(self, config: ConfigFP, y: Optional[AudioSignal],
                 data: FingerprintDatabase = None,
                 audio_id: Optional[int] = None):
        """
        A fingerprint can be either initialized from an AudioSignal class, or
        from given FingerprintDatabase (i.e. all calculations were
        already performed). This way, loading fingerprints from files, with
        need of the original audio, can be performed
        """

        self.config = config
        self.peaks = None

        # make sure only signal/data was given
        assert (y is None) != (data is None), "only y or data has to be given"

        # in case a signal was given, calculate fingerprint
        if y is not None:
            data = self._calc_fingerprint(y)

        self.data = data
        self.audio_id = audio_id

    def _calc_fingerprint(self, y: AudioSignal) -> FingerprintDatabase:

        # preliminaries
        c = self.config
        cf = self.config.fingerprint
        y_sig = y.get_signal()

        # calculate spectrum
        spec = calc_spectrum(y_sig, **c.stft_args, fs=c.fs)

        # peak-finding
        peaks = get_spec_peaks(
            spec, connectivity_mask=cf["connectivity_mask"],
            peak_neighborhood_size=cf["peak_neighborhood_size"],
            min_val_relative=cf["min_val_relative"],
            min_freq=cf["min_freq"], max_freq=cf["max_freq"],
            plot_flag=self.config.plot_flag, verbose=self.config.verbose)
        self.peaks = peaks

        # generate patterns
        patterns = generate_patterns(peaks, fan_value=cf["fan_value"],
                                     min_time_delta=cf["min_time_delta"],
                                     max_time_delta=cf["max_time_delta"],
                                     verbose=self.config.verbose)

        # convert to an efficient form for selection
        fp_database = FingerprintDatabase(patterns)

        return fp_database

    def match_fingerprint(self, fingerprint: 'Fingerprint') -> pd.DataFrame:
        """find matches of an input fingerprint to the current fingerprint"""

        data_to_be_matched = self.get_data()
        data_to_search_in = fingerprint.get_data()
        matches = data_to_be_matched.search_fingerprints(data_to_search_in)

        return matches

    def get_data(self) -> FingerprintDatabase:
        return self.data

    def get_peaks(self) -> pd.DataFrame:
        return self.peaks

    def save(self, path: Union[str, Path]):
        """save to file"""

        assert self.data is not None, "fingerprint wasn't calculated yet"
        path = Path(path)
        with open(path, 'wb') as h:
            pickle.dump(self, h)

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        """load from file"""

        path = Path(path)
        with open(path, 'rb') as h:
            obj = pickle.load(h)
        assert isinstance(obj, cls)
        return obj


class FingerprintFitter:
    """
    A class for managing the fitting process of the fingerprints of
    2 audio signals, to perform alignment estimation
    """
    def __init__(self, config: ConfigFP, ref_fp: Fingerprint,
                 target_fp: Fingerprint):
        """
        :param config: estimation config
        :param ref_fp: reference fingerprint, of the reference signal
        :param target_fp: target fingerprint, of the target signal (this signal
        will be aligned to the reference signal)
        """
        self.config = config
        self.ref_fp = ref_fp
        self.target_fp = target_fp
        self.is_fit_calculated = False
        self.fit_results = None

    def fit(self, verbose: bool = False) -> Tuple[float, float]:
        """
        :param verbose: boolean flag, whether to print messages
        :return: a tuple (offset_est, certainty_entropy), where offset_est is
        the estimated offset in seconds and certainty_entropy is a certainty
        measure of the estimation based on entropy of the yielded histogram
        (1 is maximum certainy, 0 is no certainty)
        """

        # some preliminaries
        c = self.config
        time_res = c.fit["time_res"]
        time_res_coarse = c.fit["time_res_coarse"]
        len_focus_interval = c.fit["len_focus_interval"]
        finest_res = (c.stft_args["nperseg"]-c.stft_args["noverlap"]) / c.fs
        if verbose:
            if time_res < finest_res:
                print(f"desired time_res is too low, best possible "
                      f"resolution is {finest_res} seconds")
            if time_res_coarse < finest_res:
                print(
                    f"desired time_res_coarse is too low, best possible "
                    f"resolution is {finest_res} seconds")
        time_res = max(time_res, finest_res)
        time_res_coarse = max(time_res_coarse, finest_res)

        # find fingerprint matches
        matches = self.target_fp.match_fingerprint(self.ref_fp)
        matches = matches.rename(columns={"t_abs_base": "t_abs_target",
                                          "t_abs_found": "t_abs_ref"})
        matches["diff"] = matches["t_abs_ref"] - matches["t_abs_target"]
        matches_found = matches[matches.t_abs_ref.notnull()]

        # calculate histogram to find optimal match,
        # for better performance - start from a low resolution and then
        # generate high resolution histogram around the maximal value
        d = matches_found["diff"].values
        hist_coarse, offsets_coarse = self._calc_histogram(d, time_res_coarse)

        idx = np.argmax(hist_coarse)
        t_center = offsets_coarse[idx]
        dt = len_focus_interval/2

        d_focus = d[(d >= t_center - dt) & (d <= t_center + dt)]
        hist, offsets = self._calc_histogram(d_focus, time_res)

        # find optimal offset and evaluate certainty based on entropy
        offset_est = offsets[np.argmax(hist)]
        certainty_entropy = 1 - entropy(hist_coarse/np.sum(hist_coarse),
                                        base=2) / np.log2(hist_coarse.size)
        self.is_fit_calculated = True

        # print results
        if verbose:
            print(f"Optimal offset is {offset_est} seconds")
            print(f"Entropy based certainty: {certainty_entropy}")

        # save intermediate results
        self.fit_results = {
            "matches": matches,
            "offset_est": offset_est,
            "certainty_entropy": certainty_entropy,
            "hist_coarse": hist_coarse,
            "offsets_coarse": offsets_coarse,
            "hist": hist,
            "offsets": offsets
        }

        return offset_est, certainty_entropy

    def plot_results(self):
        """plot fitting results"""

        assert self.is_fit(), "fit was not performed yet"

        r = self.fit_results
        res_coarse = r["offsets_coarse"][1] - r["offsets_coarse"][0]
        res = r["offsets"][1] - r["offsets"][0]

        fig = px.bar(x=r["offsets_coarse"], y=r["hist_coarse"])
        fig.update_xaxes(title="Offset [sec]")
        fig.update_yaxes(title="Count")
        fig.update_layout(
            title=f"Optimal Offset - Coarse Histogram "
                  f"(time resolution - {res_coarse} sec)")
        fig.show()
        fig = px.bar(x=r["offsets"], y=r["hist"])
        fig.update_xaxes(title="Offset [sec]")
        fig.update_yaxes(title="Count")
        fig.update_layout(
            title=f"Optimal Offset - High Resolution Histogram "
                  f"(time resolution - {res} sec)")
        fig.show()

    def is_fit(self) -> bool:
        return self.is_fit_calculated

    def get_results(self) -> dict:
        assert self.is_fit(), "fit was not performed yet"
        return self.fit_results

    def _calc_histogram(self, data: np.ndarray, delta: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculate histogram of a given data vector, with a defined bin width
        :param data: data vector
        :param delta: desired bin width
        :return: a tuple (hist, bins_centers), where hist contains the count
        values and bins_centers is a vector with the center of each bin
        """

        nbins = int(np.ceil((data.max() - data.min()) / delta))
        hist, bins = np.histogram(data, bins=nbins)
        bins_centers = (bins[:-1] + bins[1:]) / 2

        return hist, bins_centers

