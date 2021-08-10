from pathlib import Path

from config import ConfigFP
from wrappers import evaluate_fp_synthetic_test


def main():

    # preliminaries
    path_ref_file = Path(r"data/ref_samples.wav")
    path_test_data = Path(r"test")
    config_fp = ConfigFP()

    # evaluate fingerprints method performance using synthetic test data
    config_fp.plot_flag = False
    results_fp_simulation = \
        evaluate_fp_synthetic_test(config=config_fp,
                                   path_ref_file=path_ref_file,
                                   path_results=path_test_data, num_files=50,
                                   length_sec=0.5, min_time=5, max_time=93)


if __name__ == '__main__':
    main()
