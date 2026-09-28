import numpy as np
from itertools import product

# Objective matrices
objective1 = np.array([
    [24, 29, 18, 23],
    [33, 20, 29, 32],
    [21, 42, 12, 20],
    [25, 30, 19, 24]
])

objective2 = np.array([
    [14, 21, 18, 13],
    [24, 13, 21, 23],
    [12, 30, 9, 11],
    [13, 22, 19, 14]
])

supply = [21, 24, 18, 30]
demand = [15, 22, 26, 30]

# Generate some feasible allocations and check
target_obj1 = 1898
target_obj2 = 1207

print(f"Searching for allocation with Obj1={target_obj1}, Obj2={target_obj2}...")
print()

# Based on patterns, let's try variations of allocations
test_cases = [
    # Format: [S1->D1, S1->D2, S1->D3, S1->D4,
    #          S2->D1, S2->D2, S2->D3, S2->D4,
    #          S3->D1, S3->D2, S3->D3, S3->D4,
    #          S4->D1, S4->D2, S4->D3, S4->D4]
    [0, 0, 0, 21, 0, 22, 0, 2, 0, 0, 18, 0, 15, 0, 8, 7],
    [0, 0, 1, 20, 0, 22, 0, 2, 0, 0, 18, 0, 15, 0, 7, 8],
    [0, 0, 0, 21, 0, 22, 1, 1, 0, 0, 18, 0, 15, 0, 7, 8],
    [0, 0, 0, 21, 0, 22, 0, 2, 0, 0, 17, 1, 15, 0, 9, 6],
    [1, 0, 0, 20, 0, 22, 0, 2, 0, 0, 18, 0, 14, 0, 8, 8],
]

for idx, case in enumerate(test_cases, 1):
    alloc = np.array(case).reshape(4, 4)
    obj1_val = np.sum(alloc * objective1)
    obj2_val = np.sum(alloc * objective2)
    supply_check = alloc.sum(axis=1)
    demand_check = alloc.sum(axis=0)
    
    supply_ok = np.array_equal(supply_check, supply)
    demand_ok = np.array_equal(demand_check, demand)
    
    if obj1_val == target_obj1 or obj2_val == target_obj2 or (supply_ok and demand_ok):
        print(f"Test case {idx}:")
        print(alloc)
        print(f"  Obj1: {obj1_val}, Obj2: {obj2_val}")
        print(f"  Supply: {supply_check} {'✓' if supply_ok else '✗'}")
        print(f"  Demand: {demand_check} {'✓' if demand_ok else '✗'}")
        if obj1_val == target_obj1 and obj2_val == target_obj2:
            print("  *** MATCH FOUND! ***")
        print()
