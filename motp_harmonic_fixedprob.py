# Filename: motp_harmonic_fixedprob.py
# Implementation of the algorithm using HARMONIC MEAN instead of geometric/arithmetic mean.
# It computes harmonic mean across objectives, a fixed probability matrix,
# static row/column penalties, and iterative allocation without recalculating penalties.
# Usage: import run_algorithm from this file and call with matrices, supply, demand.

import numpy as np

def harmonic_mean_matrix(matrices):
    arrs = [np.array(m, dtype=float) for m in matrices]
    stacked = np.stack(arrs, axis=2)
    # Harmonic mean = n / sum(1/x_i)
    # Handle zeros by replacing with a very small number to avoid division by zero
    stacked_safe = np.where(stacked == 0, 1e-10, stacked)
    with np.errstate(divide='ignore', invalid='ignore'):
        hm = stacked.shape[2] / np.sum(1.0 / stacked_safe, axis=2)
    # If original had zeros, result should be zero
    hm = np.where(np.any(stacked == 0, axis=2), 0, hm)
    return hm

def fixed_probability_matrix(harm_matrix):
    chm = np.array(harm_matrix, dtype=float)
    CHMmax = np.max(chm)
    diff = CHMmax - chm
    total = np.sum(diff)
    if total == 0:
        return np.full(chm.shape, 1.0 / chm.size)
    P = diff / total
    return P

def row_col_penalties(P):
    rows, cols = P.shape
    rpen = np.zeros(rows)
    cpen = np.zeros(cols)
    for i in range(rows):
        vals = np.sort(P[i, :])[::-1]
        if len(vals) >= 2:
            rpen[i] = vals[0] - vals[1]
        elif len(vals) == 1:
            rpen[i] = vals[0]
    for j in range(cols):
        vals = np.sort(P[:, j])[::-1]
        if len(vals) >= 2:
            cpen[j] = vals[0] - vals[1]
        elif len(vals) == 1:
            cpen[j] = vals[0]
    return rpen, cpen

def select_highest_unique_penalty(rpen, cpen, active_rows, active_cols):
    rp = np.array(rpen, dtype=float)
    cp = np.array(cpen, dtype=float)
    rp[~active_rows] = -np.inf
    cp[~active_cols] = -np.inf
    max_r = np.max(rp) if rp.size>0 else -np.inf
    max_c = np.max(cp) if cp.size>0 else -np.inf
    if max_r == -np.inf and max_c == -np.inf:
        return None, None
    if max_r > max_c:
        candidates = np.where(rp == max_r)[0]
        return ('row', int(candidates[0]))
    elif max_c > max_r:
        candidates = np.where(cp == max_c)[0]
        return ('col', int(candidates[0]))
    else:  # max_r == max_c (tie)
        # When tied, prefer row over column
        row_candidates = np.where(rp == max_r)[0]
        if row_candidates.size > 0:
            return ('row', int(row_candidates[0]))
        col_candidates = np.where(cp == max_c)[0]
        return ('col', int(col_candidates[0]))

def allocate_from_line(choice, index, P, harm, supply, demand, active_rows, active_cols):
    if choice == 'row':
        i = index
        cols = np.where(active_cols)[0]
        if cols.size == 0:
            return None
        row_probs = P[i, cols]
        maxp = np.max(row_probs)
        cand_cols = cols[np.where(row_probs == maxp)[0]]
        # Among candidates with max probability, select the one with minimum cost
        best_j = cand_cols[0]
        best_cost = harm[i, best_j]
        for j in cand_cols:
            if harm[i, j] < best_cost or (harm[i, j] == best_cost and j < best_j):
                best_cost = harm[i,j]
                best_j = j
        alloc = min(supply[i], demand[best_j])
        return (i, best_j, int(alloc))
    elif choice == 'col':
        j = index
        rows = np.where(active_rows)[0]
        if rows.size == 0:
            return None
        col_probs = P[rows, j]
        maxp = np.max(col_probs)
        cand_rows = rows[np.where(col_probs == maxp)[0]]
        # Among candidates with max probability, select the one with minimum cost
        best_i = cand_rows[0]
        best_cost = harm[best_i, j]
        for i in cand_rows:
            if harm[i, j] < best_cost or (harm[i, j] == best_cost and i < best_i):
                best_cost = harm[i,j]
                best_i = i
        alloc = min(supply[best_i], demand[j])
        return (best_i, j, int(alloc))
    return None

def run_algorithm(objective_matrices, supply, demand, debug=False):
    supply = np.array(supply, dtype=int).copy()
    demand = np.array(demand, dtype=int).copy()
    harm = harmonic_mean_matrix(objective_matrices)
    P = fixed_probability_matrix(harm)
    rpen, cpen = row_col_penalties(P)
    rows, cols = harm.shape
    allocation = np.zeros((rows, cols), dtype=int)
    active_rows = supply > 0
    active_cols = demand > 0
    
    iteration = 0
    while np.any(supply > 0) and np.any(demand > 0):
        iteration += 1
        choice, idx = select_highest_unique_penalty(rpen, cpen, active_rows, active_cols)
        if choice is None:
            break
        
        if debug:
            print(f"\nIteration {iteration}:")
            print(f"  Selected: {choice} {idx} (penalty: {rpen[idx] if choice=='row' else cpen[idx]:.6f})")
            print(f"  Supply: {supply}, Demand: {demand}")
        
        sel = allocate_from_line(choice, idx, P, harm, supply, demand, active_rows, active_cols)
        if sel is None:
            if choice == 'row':
                active_rows[idx] = False
            else:
                active_cols[idx] = False
            continue
        i, j, qty = sel
        if qty <= 0:
            if supply[i] == 0:
                active_rows[i] = False
            if demand[j] == 0:
                active_cols[j] = False
            continue
        
        if debug:
            print(f"  Allocating: S{i+1} → D{j+1}: {qty} units (harmonic cost: {harm[i,j]:.2f})")
        
        allocation[i, j] += qty
        supply[i] -= qty
        demand[j] -= qty
        if supply[i] == 0:
            active_rows[i] = False
        if demand[j] == 0:
            active_cols[j] = False
    totals = []
    for mat in objective_matrices:
        mat = np.array(mat, dtype=float)
        total = int(np.sum(allocation * mat))
        totals.append(total)
    return {
        'harmonic_matrix': harm,
        'probability_matrix': P,
        'row_penalties': rpen,
        'col_penalties': cpen,
        'allocation': allocation,
        'objective_totals': totals
    }

if __name__ == "__main__":
    import pandas as pd
    import sys
    
    # Load data from CSV files
    obj1_file = "objective1_new.csv" if len(sys.argv) <= 1 else sys.argv[1]
    obj2_file = "objective2_new.csv" if len(sys.argv) <= 2 else sys.argv[2]
    obj3_file = "objective3_new.csv" if len(sys.argv) <= 3 else sys.argv[3]
    
    print(f"Loading data from {obj1_file}, {obj2_file}, and {obj3_file}...\n")
    
    # Read objective 1
    df1 = pd.read_csv(obj1_file, index_col=0)
    objective1 = df1.iloc[:-1, :-1].values.astype(float).tolist()
    supply = df1.iloc[:-1, -1].values.astype(int).tolist()
    demand = df1.iloc[-1, :-1].values.astype(int).tolist()
    
    # Read objective 2
    df2 = pd.read_csv(obj2_file, index_col=0)
    objective2 = df2.iloc[:-1, :-1].values.astype(float).tolist()
    
    # Read objective 3
    df3 = pd.read_csv(obj3_file, index_col=0)
    objective3 = df3.iloc[:-1, :-1].values.astype(float).tolist()
    
    print("Objective 1 Matrix (Cost):")
    print(np.array(objective1))
    print("\nObjective 2 Matrix (Time):")
    print(np.array(objective2))
    print("\nObjective 3 Matrix (Distance):")
    print(np.array(objective3))
    print("\nSupply:", supply)
    print("Demand:", demand)
    
    # Run algorithm with all three objective matrices
    objective_matrices = [objective1, objective2, objective3]
    
    result = run_algorithm(objective_matrices, supply, demand, debug=False)
    
    print("\n" + "="*60)
    print("RESULTS (HARMONIC MEAN)")
    print("="*60)
    print("\nHarmonic Mean Matrix:")
    print(result['harmonic_matrix'])
    print("\nProbability Matrix:")
    print(result['probability_matrix'])
    print("\nRow Penalties:", result['row_penalties'])
    print("Column Penalties:", result['col_penalties'])
    print("\nAllocation Matrix:")
    print(result['allocation'])
    print("\nAllocation Details:")
    for i in range(result['allocation'].shape[0]):
        for j in range(result['allocation'].shape[1]):
            if result['allocation'][i, j] > 0:
                cost1 = objective1[i][j]
                cost2 = objective2[i][j]
                cost3 = objective3[i][j]
                qty = result['allocation'][i, j]
                print(f"  S{i+1} → D{j+1}: {qty} units (Cost: {cost1}, Time: {cost2}, Distance: {cost3}) = Total: {qty*cost1}, {qty*cost2}, {qty*cost3}")
    print("\nObjective Totals:", result['objective_totals'])
    print("Objective 1 Total (Cost):", result['objective_totals'][0])
    print("Objective 2 Total (Time):", result['objective_totals'][1])
    print("Objective 3 Total (Distance):", result['objective_totals'][2])
