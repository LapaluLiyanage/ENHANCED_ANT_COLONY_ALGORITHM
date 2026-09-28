"""Fixed-probability MOTP heuristic using the arithmetic combiner (backwards-compatible wrapper).

The implementation now lives in the ``motp`` package; this file keeps the old
import path and command line working:

    from motp_arithmetic_fixedprob import run_algorithm
    python motp_arithmetic_fixedprob.py [objective1.csv objective2.csv ...]
"""

from motp.legacy import main, make_run_algorithm

run_algorithm = make_run_algorithm("arithmetic")

if __name__ == "__main__":
    main("arithmetic", ["objective1.csv", "objective2.csv"])
