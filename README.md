# Enhanced Ant Colony Algorithm

**Enhanced Ant Colony Algorithm Incorporating Geometric Mean for Multi-Objective Transportation Challenges**

An ant-colony-inspired heuristic for solving the **Multi-Objective Transportation Problem (MOTP)**. Instead of running a full ACO simulation, each script builds a fixed pheromone-like probability matrix from a combined-objective cost matrix, derives row/column penalties (VAM-style), and greedily allocates supply to demand along the highest-penalty row/column until the transportation table is balanced.

The core algorithm is implemented once per "combining function" used to merge the objective matrices into a single cost matrix:

| Script | Combining function | Notes |
|---|---|---|
| [`motp_geometric_fixedprob.py`](motp_geometric_fixedprob.py) | Geometric mean | The method as presented in the source paper |
| [`motp_arithmetic_fixedprob.py`](motp_arithmetic_fixedprob.py) | Arithmetic mean | Baseline comparison variant |
| [`motp_harmonic_fixedprob.py`](motp_harmonic_fixedprob.py) | Harmonic mean | Supports 3 objectives (cost, time, distance) |
| [`motp_minimum_fixedprob.py`](motp_minimum_fixedprob.py) | Element-wise minimum | Baseline comparison variant |

## How it works

1. **Combine objectives** — merge the per-objective cost matrices into a single matrix using the script's combining function (geometric/arithmetic/harmonic mean, or minimum).
2. **Build a fixed probability matrix** — `P = (max - combined) / sum(max - combined)`, giving higher "pheromone" probability to lower-cost cells.
3. **Compute row/column penalties** — for each row and column, the gap between the top two probabilities (Vogel's Approximation Method style), computed once and *not* recalculated between iterations.
4. **Allocate greedily** — repeatedly pick the row/column with the highest unique penalty, allocate to its highest-probability cell (ties broken by lowest cost), and reduce supply/demand until the table is exhausted.
5. **Report totals** — the resulting allocation matrix and the total cost per objective.

## Requirements

- Python 3.8+
- `numpy`
- `pandas` (only needed to run a script directly via its CLI, for CSV loading)

```bash
pip install numpy pandas
```

## Usage

Each `motp_*_fixedprob.py` file can be used as a library or run directly from the command line.

### As a library

```python
from motp_geometric_fixedprob import run_algorithm

objective1 = [[24, 29, 18, 23], [33, 20, 29, 32], [21, 42, 12, 20], [25, 30, 19, 24]]
objective2 = [[14, 21, 18, 13], [24, 13, 21, 23], [12, 30, 9, 11], [13, 22, 19, 14]]
supply = [21, 24, 18, 30]
demand = [15, 22, 26, 30]

result = run_algorithm([objective1, objective2], supply, demand)

print(result["allocation"])         # allocation matrix
print(result["objective_totals"])   # total cost per objective
```

### From the command line

The 2-objective scripts (`geometric`, `arithmetic`, `minimum`) read two CSV files:

```bash
python motp_geometric_fixedprob.py objective1.csv objective2.csv
```

The 3-objective script (`harmonic`) reads three:

```bash
python motp_harmonic_fixedprob.py objective1_new.csv objective2_new.csv objective3_new.csv
```

If no arguments are given, each script falls back to its default CSV filenames in the repo root.

### CSV format

Each CSV is a cost matrix with an extra `Supply` column and `Demand` row, e.g. [`objective1.csv`](objective1.csv):

```csv
"","D1","D2","D3","D4","Supply"
"S1","24","29","18","23","21"
"S2","33","20","29","32","24"
"S3","21","42","12","20","18"
"S4","25","30","19","24","30"
"Demand","15","22","26","30",""
```

Total supply must equal total demand (a balanced transportation problem).

## Sample data

- `objective1.csv` / `objective2.csv` — 4x4 balanced problem (2 objectives)
- `objective1_new.csv` / `objective2_new.csv` / `objective3_new.csv` — 3x4 balanced problem (3 objectives: cost, time, distance)

## Utility scripts

- [`search_solution.py`](search_solution.py) — brute-force search over candidate allocations to check them against target objective totals.
- [`test_allocations.py`](test_allocations.py) — scratch script for evaluating and comparing specific allocation matrices against the objective matrices.

These two are exploratory/debugging aids used while validating the algorithm's output, not part of the core library.

## License

No license has been specified for this repository.
