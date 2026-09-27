"""Fixed-probability MOTP heuristic using the minimum combiner (backwards-compatible wrapper).

The implementation now lives in the ``motp`` package; this file keeps the old
import path and command line working:

    from motp_minimum_fixedprob import run_algorithm
    python motp_minimum_fixedprob.py [objective1.csv objective2.csv ...]
"""

from motp.legacy import main, make_run_algorithm

run_algorithm = make_run_algorithm("minimum")

if __name__ == "__main__":
    main("minimum", ["objective1.csv", "objective2.csv"])
