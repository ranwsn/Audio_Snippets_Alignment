from pathlib import Path

from config import ConfigFP, ConfigCC
from wrappers import estimate_fp_folder_segments, estimate_cc_folder_segments


def main():

    # preliminaries
    path_ref_file = Path(r"data/ref_samples.wav")
    path_folder_data = Path(r"data/samples")
    path_results = Path(r"results")
    config_fp = ConfigFP()
    config_cc = ConfigCC()

    # perform estimation using fingerprints method
    print("*** Estimating with fingerprints method.. ***")
    results_fp = estimate_fp_folder_segments(config_fp,
                                             path_ref_file=path_ref_file,
                                             path_folder_data=path_folder_data,
                                             path_results=path_results,
                                             save_results_flag=True)

    # perform estimation using cross-corrleation method
    print("*** Estimating with cross-correlation method.. ***")
    results_cc = estimate_cc_folder_segments(config_cc,
                                             path_ref_file=path_ref_file,
                                             path_folder_data=path_folder_data,
                                             path_results=path_results,
                                             save_results_flag=True)


if __name__ == '__main__':
    main()
