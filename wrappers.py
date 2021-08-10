import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import soundfile as sf

from base_classes import AudioSignal, FingerprintFitter
from config import ConfigFP, ConfigCC
from signal_processing import cross_correlation


def estimate_fp_folder_segments(config: ConfigFP,
                                path_ref_file: Union[str, Path],
                                path_folder_data: Union[str, Path],
                                path_results: Optional[Union[str, Path]] = None,
                                save_results_flag: bool = True) -> pd.DataFrame:
    """
    perform alignment estimation using the fingerprints method,
    for all files in a given folder
    :param config: config for estimation
    :param path_ref_file: path of the reference file, the target files will be
    aligned according to this file
    :param path_folder_data: a path of files with segments to be aligned
    :param path_results: path to save the results
    :param save_results_flag: boolean flag, whether to save results
    :return: a dataframe with the results
    """

    # preliminaries
    path_ref_file = Path(path_ref_file)
    path_folder_data = Path(path_folder_data)
    if path_results is not None:
        path_results = Path(path_results)
        path_results.mkdir(parents=True, exist_ok=True)
    verbose = config.verbose
    plot_flag = config.plot_flag
    results = {
        "file_path": [],
        "estimated_alignment": [],
        "certainty_entropy": []
    }

    # prepare reference signal
    audio_ref = AudioSignal.from_file(path_ref_file, config=config)
    audio_ref.generate_fingerprint()
    fingerprint_ref = audio_ref.get_fingerprint()

    # estimate alignment for each target file
    target_files = path_folder_data.glob("*.wav")
    for file in target_files:

        if verbose:
            print(f"*** Analyzing file: {file} ***")

        # load audio
        audio_target = AudioSignal.from_file(file, config=config)
        if plot_flag:
            audio_target.plot()

        # calculate fingerprint
        audio_target.generate_fingerprint()
        fingerprint_target = audio_target.get_fingerprint()

        # fit fingerprints and generate estimation
        fitter = FingerprintFitter(config, fingerprint_ref, fingerprint_target)
        estimated_alignment, certainty_entropy = fitter.fit(verbose=verbose)
        if plot_flag:
            fitter.plot_results()

        # save results
        results["file_path"].append(file)
        results["estimated_alignment"].append(estimated_alignment)
        results["certainty_entropy"].append(certainty_entropy)

    # convert to df and save to file
    results = pd.DataFrame.from_dict(results)
    if save_results_flag:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results.to_csv(path_results.joinpath(timestamp+"_fp_results.csv"))
        with open(path_results.joinpath(timestamp+"_fp_config.json"), "w") as h:
            json.dump(config.dict(), h)

    return results


def evaluate_fp_synthetic_test(config: ConfigFP,
                               path_ref_file: Union[str, Path],
                               path_results: Union[str, Path],
                               num_files: int, length_sec: float,
                               min_time: float, max_time: float
                               ) -> pd.DataFrame:
    """
    Evaluate the fingerprint method for segment alignment estimation, by
    generating test data from a given reference file, and performing estimation
    on the generated segments
    :param config: estimation configuration
    :param path_ref_file: path of the reference file used for generating the
    test
    :param path_results: path to save the results
    :param num_files: number of files to generate (i.e. number of segments)
    :param length_sec: length in seconds of each generated segment
    :param min_time: the earliest time in the reference signal, that each
    segment can begin at
    :param max_time: the latest time in the reference signal, that each
    segment can end at
    :return: a dataframe with the estimation results (also saved
    in path_results)
    """

    # generate test data
    df_true = generate_test_data(path_ref_file, path_results,
                                 num_files=num_files, length_sec=length_sec,
                                 min_time=min_time, max_time=max_time)

    # perform estimation
    df_estimate = estimate_fp_folder_segments(config, path_ref_file,
                                              path_results, path_results,
                                              save_results_flag=True)

    # evaluate performance
    res = df_true.merge(df_estimate, how="left", left_on="segment_path",
                        right_on="file_path")
    res["error"] = res["true_start_time"] - res["estimated_alignment"]
    res["error_abs"] = res["error"].abs()
    error_mean = res["error_abs"].mean()

    # save and plot results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    res.to_csv(path_results.joinpath(timestamp+"_summary_fp_test.csv"))
    print(f"mean alignment error is {error_mean} seconds")
    delta = 1e-3
    nbins = int(np.ceil((res.error.max() - res.error.min()) / delta))
    px.histogram(res, x="error", title="Error Histogram [sec]",
                 nbins=nbins).show()

    return res


def generate_test_data(path_ref_file: Union[str, Path],
                       path_target_folder: Union[str, Path],
                       num_files: int, length_sec: float,
                       min_time: float, max_time: float) -> pd.DataFrame:
    """
    Generate synthetic test data from a given reference file.
    The reference file is being cut to segments in random times. These audio
    segments are being saved as wav files, and also a dataframe with each
    segment ground-truth data is being generated and saved
    :param path_ref_file: the reference file, will be used to generate segments
    :param path_target_folder: folder to save the generated data
    :param num_files: number of files to generate (i.e. number of segments)
    :param length_sec: length in seconds of each generated segment
    :param min_time: the earliest time in the reference signal, that each
    segment can begin at
    :param max_time: the latest time in the reference signal, that each
    segment can end at
    :return: a dataframe with all the ground-truth data about the generated
    segments (also being saved in path_target_folder)
    """

    # preliminaries
    path_ref_file = Path(path_ref_file)
    path_target_folder = Path(path_target_folder)
    path_target_folder.mkdir(parents=True, exist_ok=True)
    y, fs = sf.read(path_ref_file)
    length_samples = int(np.ceil(length_sec * fs))
    min_sample = int(np.ceil(min_time * fs))
    max_sample = int(np.floor(max_time * fs))
    assert length_samples < len(y)
    assert y.ndim == 1

    # generate sections
    data = {
        "ref_signal_path": [],
        "segment_path": [],
        "true_start_time": [],
        "true_start_sample": []
    }
    for j in range(num_files):
        i_end = len(y)
        while i_end >= len(y):
            i_start = np.random.randint(min_sample, max_sample)
            i_end = i_start + length_samples
        y_seg = y[i_start:i_end]
        base_name = path_ref_file.stem + "_segment"
        segment_path = path_target_folder.joinpath(base_name + str(j) + ".wav")
        sf.write(segment_path, y_seg, fs)

        data["ref_signal_path"].append(path_ref_file)
        data["segment_path"].append(segment_path)
        data["true_start_time"].append(i_start/fs)
        data["true_start_sample"].append(i_start)
    data = pd.DataFrame.from_dict(data)

    # save generated data
    data.to_csv(path_target_folder.joinpath(path_ref_file.stem + "_test.csv"))

    return data


def estimate_cc_folder_segments(config: ConfigCC,
                                path_ref_file: Union[str, Path],
                                path_folder_data: Union[str, Path],
                                path_results: Optional[Union[str, Path]] = None,
                                save_results_flag: bool = True) -> pd.DataFrame:
    """
    perform alignment estimation using the cross-correlation method,
    for all files in a given folder
    :param config: config for estimation
    :param path_ref_file: path of the reference file, the target files will be
    aligned according to this file
    :param path_folder_data: a path of files with segments to be aligned
    :param path_results: path to save the results
    :param save_results_flag: boolean flag, whether to save results
    :return: a dataframe with the results
    """

    # TODO: similar to estimate_fp_folder_segments(), can be unified

    # preliminaries
    path_ref_file = Path(path_ref_file)
    path_folder_data = Path(path_folder_data)
    if path_results is not None:
        path_results = Path(path_results)
        path_results.mkdir(parents=True, exist_ok=True)
    verbose = config.verbose
    plot_flag = config.plot_flag
    results = {
        "file_path": [],
        "estimated_alignment": []
    }

    # prepare reference signal
    x_ref, fs = sf.read(path_ref_file)

    # estimate alignment for each target file
    target_files = path_folder_data.glob("*.wav")
    for file in target_files:

        if verbose:
            print(f"*** Analyzing file: {file} ***")

        # load audio
        x_target, fs_tmp = sf.read(file)
        assert fs == fs_tmp

        # estimate offest
        estimated_alignment = \
            cross_correlation(x_ref, x_target, fs,
                              hilbert_flag=config.hilbert_flag,
                              normalize_flag=config.normalize_flag,
                              freq_range=config.freq_range, calc_offset=True,
                              plot_flag=config.plot_flag)[-1]

        # save results
        results["file_path"].append(file)
        results["estimated_alignment"].append(estimated_alignment)

    # convert to df and save to file
    results = pd.DataFrame.from_dict(results)
    if save_results_flag:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results.to_csv(path_results.joinpath(timestamp+"_cc_results.csv"))
        with open(path_results.joinpath(timestamp+"_cc_config.json"), "w") as h:
            json.dump(config.dict(), h)

    return results
