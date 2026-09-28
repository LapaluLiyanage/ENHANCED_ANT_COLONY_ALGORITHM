"""Fixed-probability MOTP heuristic using the harmonic combiner (backwards-compatible wrapper).

The implementation now lives in the ``motp`` package; this file keeps the old
import path and command line working:

    from motp_harmonic_fixedprob import run_algorithm
    python motp_harmonic_fixedprob.py [objective1.csv objective2.csv ...]
"""

from motp.legacy import main, make_run_algorithm

run_algorithm = make_run_algorithm("harmonic")

if __name__ == "__main__":
    main("harmonic", ["objective1_new.csv", "objective2_new.csv", "objective3_new.csv"])
