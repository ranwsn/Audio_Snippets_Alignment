Estimation of audio snippet alignment relative to a given reference signal, 
using audio fingerprints matching ("Shazam-like" alogrithm, based on 
shift-invariant fingerprints of the spectrum).

As a baseline method, estimation using cross-correlation was also implemented.

The main script is 'main.py', this runs both algorithms on the files in 'data'
folder.
The algorithms config can be tuned in 'config.py' (plots can be
disabled from there too by changing the flag).

The script 'test_fp_config.py' can be used to generate synthetic test from a
reference signal, and evaluate the fingerprints method on it (good for
hyperparameters tuning).

I used Python 3.8, the requirements are given in a file.

Note that part of the algorithm implementation is based on the code in this repository:
https://github.com/worldveil/dejavu
