from pydantic import BaseModel


class ConfigFP(BaseModel):
    """ config for alignment estimation using fingerprints """

    verbose: bool = True
    plot_flag: bool = True
    fs: int = 16000
    stft_args: dict = {
        "nperseg": 1024,
        "noverlap": int(0.9*1024),
        "window": "hann"
    }
    fingerprint: dict = {
        "connectivity_mask": 2,
        "min_val_relative": -35,
        "min_freq": 50,
        "max_freq": 5000,
        "peak_neighborhood_size": 2,
        "fan_value": 10,
        "min_time_delta": 0,
        "max_time_delta": 0.5
    }
    fit: dict = {
        "time_res": 0.001,
        "time_res_coarse": 0.08,
        "len_focus_interval": 5
    }


class ConfigCC(BaseModel):
    """ config for alignment estimation using cross-correlation """
    verbose: bool = True
    plot_flag: bool = True
    hilbert_flag: bool = True
    normalize_flag: bool = True
    freq_range: list = [200, 4000]
