import numpy as np

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

# Current allocation
current = np.array([
    [15,  0,  6,  0],
    [ 0, 22,  2,  0],
    [ 0,  0, 18,  0],
    [ 0,  0,  0, 30]
])

print("Current allocation:")
print(current)
print(f"Objective 1: {np.sum(current * objective1)}")
print(f"Objective 2: {np.sum(current * objective2)}")
print()

# Try to find allocation that gives 1898, 1207
# Let's try different possibilities
print("Testing alternative allocations...")
print()

# Try: allocate S4->D1 instead of S1->D1
alt1 = np.array([
    [ 0,  0, 21,  0],
    [ 0, 22,  2,  0],
    [ 0,  0, 18,  0],
    [15,  0,  0, 15]
])
print("Alternative 1: S1->D3:21, S4->D1:15, S4->D4:15")
print(f"Objective 1: {np.sum(alt1 * objective1)}")
print(f"Objective 2: {np.sum(alt1 * objective2)}")
print()

# Try: S1->D4, S2->D2, S3->D3, S4->D1+D3
alt2 = np.array([
    [ 0,  0,  0, 21],
    [ 0, 22,  2,  0],
    [ 0,  0, 18,  0],
    [15,  0,  6,  9]
])
print("Alternative 2: S1->D4:21, S2->D2:22+D3:2, S3->D3:18, S4->D1:15+D3:6+D4:9")
print(f"Objective 1: {np.sum(alt2 * objective1)}")
print(f"Objective 2: {np.sum(alt2 * objective2)}")
print()

# Try different combinations
alt3 = np.array([
    [ 0,  0,  0, 21],
    [ 0, 22,  0,  2],
    [ 0,  0, 18,  0],
    [15,  0,  8,  7]
])
print("Alternative 3: S1->D4:21, S2->D2:22+D4:2, S3->D3:18, S4->D1:15+D3:8+D4:7")
print(f"Objective 1: {np.sum(alt3 * objective1)}")
print(f"Objective 2: {np.sum(alt3 * objective2)}")
print()

# Check supply/demand constraints
print("Checking alt3 constraints:")
print(f"Supply check: {alt3.sum(axis=1)} should be [21, 24, 18, 30]")
print(f"Demand check: {alt3.sum(axis=0)} should be [15, 22, 26, 30]")
print()

# More alternatives around 1898
alt4 = np.array([
    [ 0,  0,  2, 19],
    [ 0, 22,  0,  2],
    [ 0,  0, 18,  0],
    [15,  0,  6,  9]
])
print("Alternative 4: S1->D3:2+D4:19, S2->D2:22+D4:2, S3->D3:18, S4->D1:15+D3:6+D4:9")
print(f"Objective 1: {np.sum(alt4 * objective1)}")
print(f"Objective 2: {np.sum(alt4 * objective2)}")
print(f"Supply: {alt4.sum(axis=1)}, Demand: {alt4.sum(axis=0)}")
print()

alt5 = np.array([
    [ 0,  0,  6, 15],
    [ 0, 22,  0,  2],
    [ 0,  0, 18,  0],
    [15,  0,  2, 13]
])
print("Alternative 5: S1->D3:6+D4:15, S2->D2:22+D4:2, S3->D3:18, S4->D1:15+D3:2+D4:13")
print(f"Objective 1: {np.sum(alt5 * objective1)}")
print(f"Objective 2: {np.sum(alt5 * objective2)}")
print(f"Supply: {alt5.sum(axis=1)}, Demand: {alt5.sum(axis=0)}")
print()

alt6 = np.array([
    [ 6,  0,  0, 15],
    [ 0, 22,  2,  0],
    [ 0,  0, 18,  0],
    [ 9,  0,  6, 15]
])
print("Alternative 6: S1->D1:6+D4:15, S2->D2:22+D3:2, S3->D3:18, S4->D1:9+D3:6+D4:15")
print(f"Objective 1: {np.sum(alt6 * objective1)}")
print(f"Objective 2: {np.sum(alt6 * objective2)}")
print(f"Supply: {alt6.sum(axis=1)}, Demand: {alt6.sum(axis=0)}")
print()
