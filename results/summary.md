# Benchmark summary

60 instances, sizes [(5, 5), (10, 10), (15, 20)], objectives [2, 3], correlations [0.0, 0.5]; 3 runs per stochastic method, averaged per instance.

## gap (lower is better)

Friedman chi2 = 515.1, p = 1.97e-103

| method | mean | median | avg rank | vs FP-geometric: better/worse | Wilcoxon p (Holm) |
|---|---|---|---|---|---|
| Exact LP (weighted sums) | 0.3049 | 0.1900 | 1.77 | 60/0 | 1.8e-10 |
| MOACO+LS (FP-seeded) | 0.3068 | 0.1908 | 2.42 | 60/0 | 1.8e-10 |
| MOACO+LS | 0.3071 | 0.1910 | 2.56 | 60/0 | 1.8e-10 |
| MOACO | 0.3280 | 0.2120 | 3.79 | 60/0 | 1.8e-10 |
| VAM-arithmetic-norm | 0.3651 | 0.2462 | 5.51 | 51/6 | 4.3e-08 |
| FP-arithmetic | 0.4336 | 0.3055 | 7.53 | 33/23 | 1.2e-01 |
| FP-arithmetic-norm | 0.4356 | 0.2997 | 7.69 | 32/25 | 1.6e-01 |
| FP-geometric | 0.4615 | 0.3186 | 8.28 | - | - |
| FP-harmonic | 0.5030 | 0.3499 | 9.08 | 19/31 | 3.8e-02 |
| FP-harmonic-norm | 0.5042 | 0.3446 | 9.22 | 18/35 | 3.8e-02 |
| FP-minimum-norm | 0.5371 | 0.3816 | 10.01 | 14/43 | 3.7e-04 |
| FP-minimum | 0.5404 | 0.3770 | 10.12 | 14/44 | 2.7e-04 |

## hv (higher is better)

Friedman chi2 = 516.8, p = 8.57e-104

| method | mean | median | avg rank | vs FP-geometric: better/worse | Wilcoxon p (Holm) |
|---|---|---|---|---|---|
| MOACO+LS (FP-seeded) | 0.9617 | 0.9819 | 1.97 | 60/0 | 1.8e-10 |
| MOACO+LS | 0.9612 | 0.9812 | 2.08 | 60/0 | 1.8e-10 |
| Exact LP (weighted sums) | 0.9425 | 0.9604 | 2.44 | 60/0 | 1.8e-10 |
| MOACO | 0.8798 | 0.9004 | 3.52 | 60/0 | 1.8e-10 |
| VAM-arithmetic-norm | 0.4158 | 0.4428 | 6.01 | 50/6 | 7.0e-09 |
| FP-arithmetic | 0.2738 | 0.2790 | 7.74 | 34/20 | 4.7e-02 |
| FP-arithmetic-norm | 0.2712 | 0.2655 | 7.77 | 33/21 | 4.7e-02 |
| FP-geometric | 0.2223 | 0.1960 | 8.60 | - | - |
| FP-harmonic-norm | 0.1943 | 0.1857 | 9.07 | 19/30 | 1.1e-01 |
| FP-harmonic | 0.1899 | 0.1807 | 9.22 | 18/29 | 1.1e-01 |
| FP-minimum-norm | 0.1622 | 0.1224 | 9.78 | 19/37 | 2.0e-02 |
| FP-minimum | 0.1572 | 0.1290 | 9.80 | 19/37 | 1.2e-02 |

## Mean runtime per instance (s)

| method | seconds |
|---|---|
| FP-geometric | 0.0007 |
| FP-arithmetic | 0.0006 |
| FP-harmonic | 0.0005 |
| FP-minimum | 0.0005 |
| FP-arithmetic-norm | 0.0005 |
| FP-harmonic-norm | 0.0005 |
| FP-minimum-norm | 0.0005 |
| VAM-arithmetic-norm | 0.0020 |
| MOACO | 0.5354 |
| MOACO+LS | 0.6354 |
| MOACO+LS (FP-seeded) | 0.6388 |
| Exact LP (weighted sums) | 0.4755 |
